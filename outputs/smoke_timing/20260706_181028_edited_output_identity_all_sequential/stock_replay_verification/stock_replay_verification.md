# FACE3 stock replay verification

- run root: `outputs/smoke_timing/20260706_181028_edited_output_identity_all_sequential`
- cases verified: 4
- wrapper best edits matching stock replay: 0 / 4
- wrapper final edits matching stock replay: 4 / 4
- best perturbed edits disruptive under stock replay: 0 / 4
- final perturbed edits disruptive under stock replay: 0 / 4

Interpretation: if wrapper-vs-stock SSIM is high, the saved differentiable edit is reproduced by the normal stock pipeline for that saved perturbed image.

## face_002__add_black_sunglasses

- prompt: add black sunglasses
- stock vs saved best-wrapper SSIM: 0.314547
- stock clean vs stock best SSIM: 0.963286
- stock vs saved final-wrapper SSIM: 0.997098
- stock clean vs stock final SSIM: 0.946919
- sheet: `outputs/smoke_timing/20260706_181028_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_002__add_black_sunglasses/stock_replay_verification/stock_replay_sheet.png`

## face_002__add_headphones

- prompt: add headphones
- stock vs saved best-wrapper SSIM: 0.333841
- stock clean vs stock best SSIM: 0.934411
- stock vs saved final-wrapper SSIM: 0.995561
- stock clean vs stock final SSIM: 0.918761
- sheet: `outputs/smoke_timing/20260706_181028_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_002__add_headphones/stock_replay_verification/stock_replay_sheet.png`

## face_005__add_black_sunglasses

- prompt: add black sunglasses
- stock vs saved best-wrapper SSIM: 0.235100
- stock clean vs stock best SSIM: 0.972241
- stock vs saved final-wrapper SSIM: 0.996813
- stock clean vs stock final SSIM: 0.959794
- sheet: `outputs/smoke_timing/20260706_181028_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_005__add_black_sunglasses/stock_replay_verification/stock_replay_sheet.png`

## face_005__add_headphones

- prompt: add headphones
- stock vs saved best-wrapper SSIM: 0.366654
- stock clean vs stock best SSIM: 0.957499
- stock vs saved final-wrapper SSIM: 0.996339
- stock clean vs stock final SSIM: 0.952224
- sheet: `outputs/smoke_timing/20260706_181028_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_005__add_headphones/stock_replay_verification/stock_replay_sheet.png`
