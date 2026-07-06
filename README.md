# FACE3

FACE3 is a compact white-box edited-output identity optimization experiment.

The optimized scalar is:

```text
clean_edit      = InstructPix2Pix(original, prompt, fixed seed/settings)
perturbed_input = T_theta(original)
perturbed_edit  = InstructPix2Pix(perturbed_input, prompt, same seed/settings)

Z = cosine_similarity(ArcFace(clean_edit), ArcFace(perturbed_edit))
loss = Z
```

The optimizer minimizes `loss = Z`, so it tries to reduce ArcFace cosine similarity between the clean edited output and the perturbed edited output.

## Is backpropagation through InstructPix2Pix + ArcFace possible?

Yes, as long as the optimization does not call the normal no-grad/PIL diffusers pipeline. FACE3 uses a differentiable InstructPix2Pix mini-forward:

```text
ArcFace
← perturbed edited image
← differentiable InstructPix2Pix denoising loop
← perturbed input image
← differentiable perturbation module
← perturbation parameters
```

InstructPix2Pix weights and ArcFace weights are frozen. Only perturbation parameters are optimized with Adam.

This is much heavier than FACE/FACE2 because every optimization iteration unrolls an image edit. Start with smoke tests and low `--edit-steps` first.

## Perturbation modules

FACE3 uses the same perturbation module family as FACE2:

- TPS / Thin Plate Spline
- Delaunay / fixed-topology piecewise affine
- Rolling shutter
- DCT-domain image frequency perturbation
- FFT phase module

TPS, Delaunay, and rolling shutter produce spatial displacement fields. These fields are summed and applied through `grid_sample`.

The active DCT module performs blockwise image DCT coefficient changes and reconstructs the image with inverse DCT. It is not a coordinate flow field.

All enabled perturbation parameters are optimized jointly. After every optimizer step, parameters are projected/clamped using `configs/geometry_default.json`.

## Models

FACE3 uses:

- `timbrooks/instruct-pix2pix` for the differentiable edited-output path
- frozen ArcFace iResNet-100 for identity embeddings

Expected ArcFace checkpoint path:

```text
models/arcface/iresnet100.pth
```

Checkpoint files are ignored by Git and should not be committed.

## Cases

FACE3 keeps the same four cases:

- `face_002` + `add black sunglasses`
- `face_002` + `add headphones`
- `face_005` + `add black sunglasses`
- `face_005` + `add headphones`

The prompt is part of the objective because it conditions both clean and perturbed InstructPix2Pix edits.

## Outputs

Each run saves:

```text
history.csv
history.jsonl
config_resolved.json
summary.json
DONE.json or FAILED.json
checkpoints/
original.png
perturbed_best.png
perturbed_final.png
input_difference_best.png
input_difference_final.png
combined_flow_best.png
combined_flow_final.png
dct_only_perturbed.png
dct_only_difference.png
dct_only_difference_x10.png
dct_gain_heatmap.png
dct_frequency_mask.png
dct_spectrum_before.png
dct_spectrum_after.png
dct_spectrum_difference.png
original_edited.png
perturbed_best_edited.png
perturbed_final_edited.png
comparison_sheet.png
```

FACE3 intentionally does not save `theta_final.pt` or `theta_best.pt` by default, to avoid accidentally pushing replay tensors.

## Smoke-first workflow

Use the A6000 environment from MAT:

```bash
cd /home/interns/Desktop/face3
git pull origin main

bash scripts/download_arcface_checkpoint_a6000.sh /home/interns/Desktop/face3

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face3.scripts.smoke_identity \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face3/models/arcface/iresnet100.pth \
  --iters 1 \
  --edit-steps 2 \
  --geometry-config configs/geometry_default.json
```

Then timing:

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face3.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face3/models/arcface/iresnet100.pth \
  --iters 1 \
  --edit-steps 2 \
  --quick \
  --geometry-config configs/geometry_default.json
```

If quick smoke works, run all-case smoke:

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face3.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face3/models/arcface/iresnet100.pth \
  --iters 1 \
  --edit-steps 2 \
  --all-cases \
  --geometry-config configs/geometry_default.json
```

Do not jump to a long run until the smoke timing is known. Increase `--edit-steps` only if memory/time looks safe.
