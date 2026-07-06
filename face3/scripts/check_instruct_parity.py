"""Compare stock InstructPix2Pix output against FACE3 differentiable wrapper."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from face3.core.cases import Case, resolve_image_path
from face3.core.image_metrics import image_metrics, pil_to_tensor, save_sheet, tensor_to_pil
from face3.models.differentiable_instruct import DifferentiableInstructPix2Pix, DifferentiableInstructSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check stock-vs-differentiable InstructPix2Pix parity.")
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--face-id", default="face_002")
    parser.add_argument("--prompt", default="add black sunglasses")
    parser.add_argument("--seed", type=int, default=None, help="Default uses the FACE case seed.")
    parser.add_argument("--edit-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--editor-dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--output-root", default="outputs/instruct_parity")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]


@torch.inference_mode()
def stock_edit(image: Image.Image, prompt: str, seed: int, args: argparse.Namespace, device: torch.device) -> Image.Image:
    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=_dtype(args.editor_dtype),
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=args.edit_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        generator=generator,
    )
    return result.images[0].convert("RGB")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case = Case(args.face_id, args.prompt)
    seed = case.seed if args.seed is None else int(args.seed)
    image_path = resolve_image_path(Path(args.mat_root), args.face_id)
    image = Image.open(image_path).convert("RGB")
    output_root = Path(args.output_root) / f"{args.face_id}__{case.slug}__seed_{seed}__steps_{args.edit_steps}"
    output_root.mkdir(parents=True, exist_ok=True)
    image.save(output_root / "original.png")

    stock = stock_edit(image, args.prompt, seed, args, device)
    stock.save(output_root / "stock_pipeline_edit.png")

    settings = DifferentiableInstructSettings(
        torch_dtype=args.editor_dtype,
        num_inference_steps=args.edit_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        enable_gradient_checkpointing=True,
    )
    wrapper = DifferentiableInstructPix2Pix(device, settings=settings)
    tensor = pil_to_tensor(image, device)
    with torch.no_grad():
        diff = tensor_to_pil(wrapper.edit_tensor(tensor, args.prompt, seed))
    diff.save(output_root / "differentiable_wrapper_edit.png")

    metrics = image_metrics(stock, diff)
    (output_root / "metrics.json").write_text(
        __import__("json").dumps(
            {
                "face_id": args.face_id,
                "prompt": args.prompt,
                "seed": seed,
                "edit_steps": args.edit_steps,
                "guidance_scale": args.guidance_scale,
                "image_guidance_scale": args.image_guidance_scale,
                "stock_vs_differentiable": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_sheet(
        output_root / "parity_sheet.png",
        [
            ("Original", image),
            ("Stock pipe", stock),
            ("Differentiable wrapper", diff),
        ],
    )
    print(f"[face3-parity] wrote {output_root}")
    print(f"[face3-parity] stock_vs_differentiable: {metrics}")


if __name__ == "__main__":
    main()
