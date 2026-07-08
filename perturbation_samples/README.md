# FACE3 perturbation samples

These samples are model-free visual checks for individual perturbation families. They do not run InstructPix2Pix or ArcFace.

The optimizer config keeps the new components disabled by default. Enable them in `configs/geometry_default.json` when you want them active in a main run.

## Perturbation notes

### Polar coordinate shifts / perturbations

Implemented as radial expansion/compression plus radius-dependent twist around the image center. This is a direct differentiable image-plane coordinate warp.

### B-spline / Bezier-style free-form deformation

Implemented as a differentiable control-grid free-form deformation. This is the practical raster-image version of perturbing spline/curve control parameters; it does not vectorize the whole photo into literal Bezier curves.

### Lens barrel distortion

Implemented as radial lens distortion using a polynomial radius factor. It bends straight structures outward/inward depending on sampling convention and coefficient sign.

### Lens pincushion distortion

Implemented with the opposite radial coefficient sign from barrel distortion. It is the complementary radial lens family.

### Möbius transform

Implemented on normalized image coordinates as a small complex-plane linear-fractional warp z'=(z+b)/(cz+1). This is differentiable and conformal away from singularities, with limits to avoid poles inside the image.

### Laplacian-smoothed deformation

Laplacian smoothing itself is a mesh/field regularizer, not a unique warp. Here it is implemented as a trainable low-resolution displacement field repeatedly diffused by a discrete Laplacian-like neighbor average.

### Geodesic-inspired deformation

The referenced Lu/Jain 3D face method is for 2.5D/3D facial surfaces. FACE3 only has RGB images, so this is a 2D surrogate using an elliptical face metric with radial/tangential flow.

### Differential-geometry-inspired surface warp

Differential geometry of surfaces is a 3D/surface theory, not directly a raster warp. This implementation treats a scalar height map h(x,y) as a simple Monge surface and uses its gradient as a coordinate displacement.

## Reference basis

- Generic image warps are implemented as inverse coordinate maps sampled with bilinear interpolation, matching the standard remap/grid-sampling model.
- Lens barrel and pincushion use radial camera-distortion style coordinate factors.
- The B-spline/Bezier-style implementation uses a free-form deformation control grid, the practical raster-image analogue of perturbing spline control parameters.
- Möbius uses the complex linear-fractional form `(a z + b) / (c z + d)`, restricted near identity for stability.
- Laplacian smoothing is treated as displacement-field diffusion because Laplacian methods are normally mesh/field editing tools.
- Geodesic and differential-surface items require actual 3D/surface data for literal implementations, so FACE3 provides RGB image-plane surrogates.

## Generated inputs

```json
{
  "images": {
    "face_002": "C:\\Users\\parth\\Desktop\\experiment\\mat\\data\\face_002\\instruct_512.png",
    "face_005": "C:\\Users\\parth\\Desktop\\experiment\\mat\\data\\face_005\\instruct_512.png"
  },
  "perturbations": {
    "polar": {
      "title": "Polar coordinate shifts / perturbations",
      "note": "Implemented as radial expansion/compression plus radius-dependent twist around the image center. This is a direct differentiable image-plane coordinate warp."
    },
    "bspline_bezier_ffd": {
      "title": "B-spline / Bezier-style free-form deformation",
      "note": "Implemented as a differentiable control-grid free-form deformation. This is the practical raster-image version of perturbing spline/curve control parameters; it does not vectorize the whole photo into literal Bezier curves."
    },
    "lens_barrel": {
      "title": "Lens barrel distortion",
      "note": "Implemented as radial lens distortion using a polynomial radius factor. It bends straight structures outward/inward depending on sampling convention and coefficient sign."
    },
    "lens_pincushion": {
      "title": "Lens pincushion distortion",
      "note": "Implemented with the opposite radial coefficient sign from barrel distortion. It is the complementary radial lens family."
    },
    "mobius": {
      "title": "M\u00f6bius transform",
      "note": "Implemented on normalized image coordinates as a small complex-plane linear-fractional warp z'=(z+b)/(cz+1). This is differentiable and conformal away from singularities, with limits to avoid poles inside the image."
    },
    "laplacian": {
      "title": "Laplacian-smoothed deformation",
      "note": "Laplacian smoothing itself is a mesh/field regularizer, not a unique warp. Here it is implemented as a trainable low-resolution displacement field repeatedly diffused by a discrete Laplacian-like neighbor average."
    },
    "geodesic": {
      "title": "Geodesic-inspired deformation",
      "note": "The referenced Lu/Jain 3D face method is for 2.5D/3D facial surfaces. FACE3 only has RGB images, so this is a 2D surrogate using an elliptical face metric with radial/tangential flow."
    },
    "differential_surface": {
      "title": "Differential-geometry-inspired surface warp",
      "note": "Differential geometry of surfaces is a 3D/surface theory, not directly a raster warp. This implementation treats a scalar height map h(x,y) as a simple Monge surface and uses its gradient as a coordinate displacement."
    }
  },
  "levels": [
    0.25,
    0.5,
    0.75,
    1.0
  ]
}
```
