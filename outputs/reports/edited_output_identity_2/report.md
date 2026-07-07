# FACE3: Edited-output ArcFace White-box Optimization

Differentiable InstructPix2Pix edit identity results with geometric perturbations

FACE3 optimizes `Z = cosine_similarity(ArcFace(original_edit), ArcFace(perturbed_edit))` with `loss = Z`. DCT is reported as an image-frequency coefficient perturbation, not a spatial flow.

## Image strips

### face_002 / add black sunglasses

![strip](strips/face_face_002_add_black_sunglasses.png)

## Graphs

### Z vs iteration

![Z vs iteration](graphs/z_vs_iteration.png)

### Loss vs iteration

![Loss vs iteration](graphs/loss_vs_iteration.png)

### PSNR to original vs iteration

![PSNR to original vs iteration](graphs/psnr_vs_iteration.png)

### SSIM to original vs iteration

![SSIM to original vs iteration](graphs/ssim_vs_iteration.png)

### Geometry component diagnostics vs iteration

![Geometry component diagnostics vs iteration](graphs/geometry_component_diagnostics_vs_iteration.png)
