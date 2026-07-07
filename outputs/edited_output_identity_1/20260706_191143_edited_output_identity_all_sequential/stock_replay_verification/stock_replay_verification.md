# FACE3 stock replay verification

- run root: `outputs/edited_output_identity_1/20260706_191143_edited_output_identity_all_sequential`
- cases verified: 4
- wrapper best edits matching stock replay: 0 / 4
- wrapper final edits matching stock replay: 3 / 4
- best perturbed edits disruptive under stock replay: 0 / 4
- final perturbed edits disruptive under stock replay: 0 / 4

Interpretation: if wrapper-vs-stock SSIM is high, the saved differentiable edit is reproduced by the normal stock pipeline for that saved perturbed image.

## face_002__add_black_sunglasses

- prompt: add black sunglasses
- stock vs saved best-wrapper SSIM: 0.000000
- stock clean vs stock best SSIM: 0.791091
- stock vs saved final-wrapper SSIM: 0.980564
- stock clean vs stock final SSIM: 0.773095
- sheet: `outputs/edited_output_identity_1/20260706_191143_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_002__add_black_sunglasses/stock_replay_verification/stock_replay_sheet.png`

## face_002__add_headphones

- prompt: add headphones
- stock vs saved best-wrapper SSIM: 0.000000
- stock clean vs stock best SSIM: 0.771998
- stock vs saved final-wrapper SSIM: 0.982324
- stock clean vs stock final SSIM: 0.778146
- sheet: `outputs/edited_output_identity_1/20260706_191143_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_002__add_headphones/stock_replay_verification/stock_replay_sheet.png`

## face_005__add_black_sunglasses

- prompt: add black sunglasses
- stock vs saved best-wrapper SSIM: 0.000000
- stock clean vs stock best SSIM: 0.827276
- stock vs saved final-wrapper SSIM: 0.973999
- stock clean vs stock final SSIM: 0.814272
- sheet: `outputs/edited_output_identity_1/20260706_191143_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_005__add_black_sunglasses/stock_replay_verification/stock_replay_sheet.png`

## face_005__add_headphones

- prompt: add headphones
- stock vs saved best-wrapper SSIM: 0.000000
- stock clean vs stock best SSIM: 0.807947
- stock vs saved final-wrapper SSIM: 0.984948
- stock clean vs stock final SSIM: 0.778889
- sheet: `outputs/edited_output_identity_1/20260706_191143_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100/face_005__add_headphones/stock_replay_verification/stock_replay_sheet.png`
