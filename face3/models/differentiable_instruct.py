"""Differentiable InstructPix2Pix edit forward for FACE3.

The stock diffusers pipeline is intentionally inference-oriented and its
``__call__`` method is decorated with ``torch.no_grad`` in current releases.
FACE3 needs gradients from edited output identity back into the input-image
perturbation, so this module reuses the loaded pipeline modules but runs the
latent denoising loop directly under autograd.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class DifferentiableInstructSettings:
    model_id: str = "timbrooks/instruct-pix2pix"
    torch_dtype: str = "float16"
    num_inference_steps: int = 4
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    eta: float = 0.0
    enable_gradient_checkpointing: bool = False
    enable_vae_slicing: bool = True


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return aliases[name.lower()]


class DifferentiableInstructPix2Pix:
    """Small autograd-friendly wrapper around InstructPix2Pix modules."""

    def __init__(self, device: torch.device, settings: DifferentiableInstructSettings | None = None) -> None:
        self.device = device
        self.settings = settings or DifferentiableInstructSettings()
        self.dtype = _dtype(self.settings.torch_dtype)
        self.pipe = self._load_pipeline()
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = int(getattr(self.pipe, "vae_scale_factor", 8))
        self._prompt_cache: dict[tuple[str, str], torch.Tensor] = {}

    def _load_pipeline(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for module_name in ("vae", "text_encoder", "unet"):
            module = getattr(pipe, module_name, None)
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if self.settings.enable_vae_slicing and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if self.settings.enable_gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
            pipe.unet.enable_gradient_checkpointing()
        return pipe

    def metadata(self) -> dict[str, Any]:
        return {
            "editor": self.settings.model_id,
            "differentiable_editor": True,
            "stock_pipeline_call_used_for_gradient": False,
            "settings": asdict(self.settings),
            "dtype": str(self.dtype).replace("torch.", ""),
            "vae_scale_factor": self.vae_scale_factor,
        }

    @torch.no_grad()
    def prompt_embeds(self, prompt: str, negative_prompt: str = "") -> torch.Tensor:
        key = (prompt, negative_prompt)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached
        max_length = self.tokenizer.model_max_length
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_inputs = self.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        negative_embeds = self.text_encoder(negative_inputs.input_ids.to(self.device))[0]
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        negative_embeds = negative_embeds.to(device=self.device, dtype=self.dtype)
        # Diffusers InstructPix2Pix orders classifier-free branches as
        # [text, image-conditional negative, unconditional negative].
        embeds = torch.cat([prompt_embeds, negative_embeds, negative_embeds], dim=0).detach()
        self._prompt_cache[key] = embeds
        return embeds

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(0, 1)
        height, width = image.shape[-2:]
        new_height = height - height % self.vae_scale_factor
        new_width = width - width % self.vae_scale_factor
        if (new_height, new_width) != (height, width):
            image = F.interpolate(image, size=(new_height, new_width), mode="bilinear", align_corners=False)
        return image.to(device=self.device, dtype=self.dtype) * 2.0 - 1.0

    def _encode_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        image = self._preprocess_image(image)
        latent_dist = self.vae.encode(image).latent_dist
        if hasattr(latent_dist, "mode"):
            latents = latent_dist.mode()
        else:
            latents = latent_dist.mean
        latents = latents * self.vae.config.scaling_factor
        zeros = torch.zeros_like(latents)
        return torch.cat([latents, latents, zeros], dim=0)

    def _initial_latents(self, image: torch.Tensor, seed: int) -> torch.Tensor:
        height, width = image.shape[-2:]
        height = height - height % self.vae_scale_factor
        width = width - width % self.vae_scale_factor
        channels = int(self.vae.config.latent_channels)
        shape = (1, channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        return latents * self.scheduler.init_noise_sigma

    def _extra_step_kwargs(self, seed: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        try:
            import inspect

            parameters = inspect.signature(self.scheduler.step).parameters
            if "eta" in parameters:
                kwargs["eta"] = self.settings.eta
            if "generator" in parameters:
                kwargs["generator"] = torch.Generator(device=self.device).manual_seed(int(seed))
        except Exception:
            if self.settings.eta:
                kwargs["eta"] = self.settings.eta
        return kwargs

    def edit_tensor(self, image: torch.Tensor, prompt: str, seed: int) -> torch.Tensor:
        """Return edited RGB tensor in [0, 1], preserving autograd to ``image``."""
        prompt_embeds = self.prompt_embeds(prompt)
        self.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
        image_latents = self._encode_image_latents(image)
        latents = self._initial_latents(image, seed)
        extra_step_kwargs = self._extra_step_kwargs(seed)
        for timestep in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 3, dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            model_input = torch.cat([latent_model_input, image_latents], dim=1)
            noise_pred = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + float(self.settings.guidance_scale) * (noise_pred_text - noise_pred_image)
                + float(self.settings.image_guidance_scale) * (noise_pred_image - noise_pred_uncond)
            )
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False, **extra_step_kwargs)[0]
        decoded = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        return (decoded / 2.0 + 0.5).clamp(0, 1).float()
