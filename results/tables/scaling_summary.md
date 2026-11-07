# Scale-up: 120 vs 300 images (held-out test split, 95% bootstrap CIs over images)

## Win-rate vs blur-matched control

| Metric | before (120 images, 60 test) | after (300 images, 150 test) | new test images only (90) |
|---|---|---|---|
| PSNR | 0.52 [0.45, 0.60] | 0.50 [0.45, 0.55] | 0.49 [0.42, 0.54] |
| SSIM | 0.23 [0.13, 0.34] | 0.17 [0.12, 0.23] | 0.14 [0.09, 0.20] |
| LPIPS | 0.02 [0.00, 0.04] | 0.02 [0.01, 0.04] | 0.03 [0.00, 0.05] |
| DISTS | 0.24 [0.16, 0.32] | 0.22 [0.17, 0.27] | 0.21 [0.15, 0.27] |
| SHAFE-ResNet (official) | 0.84 [0.77, 0.91] | 0.82 [0.78, 0.87] | 0.82 [0.75, 0.87] |
| SHAFE-DINO | 0.77 [0.67, 0.86] | 0.74 [0.69, 0.80] | 0.72 [0.66, 0.79] |
| SemRestore (ours) | 0.98 [0.95, 0.99] | 0.99 [0.98, 1.00] | 1.00 [0.99, 1.00] |

## Win-rate vs noise-matched control

| Metric | before (120 images, 60 test) | after (300 images, 150 test) | new test images only (90) |
|---|---|---|---|
| PSNR | 0.54 [0.47, 0.61] | 0.50 [0.46, 0.54] | 0.48 [0.42, 0.53] |
| SSIM | 0.03 [0.00, 0.07] | 0.05 [0.02, 0.07] | 0.06 [0.02, 0.10] |
| LPIPS | 0.09 [0.04, 0.13] | 0.09 [0.06, 0.12] | 0.09 [0.05, 0.14] |
| DISTS | 0.28 [0.20, 0.35] | 0.30 [0.25, 0.35] | 0.31 [0.24, 0.37] |
| SHAFE-ResNet (official) | 0.95 [0.91, 0.98] | 0.94 [0.91, 0.96] | 0.93 [0.89, 0.97] |
| SHAFE-DINO | 0.70 [0.63, 0.78] | 0.74 [0.69, 0.79] | 0.77 [0.70, 0.83] |
| SemRestore (ours) | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |

## FNR at the dev-frozen 5% FPR threshold (lower is better)

| Metric | before (120 images, 60 test) | after (300 images, 150 test) | new test images only (90) |
|---|---|---|---|
| PSNR | 0.90 [0.85, 0.95] | 0.93 [0.90, 0.95] | 0.94 [0.91, 0.97] |
| SSIM | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] |
| LPIPS | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| DISTS | 0.98 [0.96, 1.00] | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] |
| SHAFE-ResNet (official) | 0.79 [0.72, 0.85] | 0.77 [0.73, 0.81] | 0.77 [0.72, 0.82] |
| SHAFE-DINO | 0.54 [0.47, 0.60] | 0.56 [0.53, 0.60] | 0.58 [0.53, 0.62] |
| SemRestore (ours) | 0.24 [0.18, 0.31] | 0.22 [0.19, 0.27] | 0.22 [0.17, 0.27] |

## AUROC: addition

| Metric | before (120 images, 60 test) | after (300 images, 150 test) | new test images only (90) |
|---|---|---|---|
| SHAFE-DINO | 1.00 [0.98, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.97, 1.00] |
| SemRestore (ours) | 0.96 [0.91, 1.00] | 0.99 [0.97, 1.00] | 0.99 [0.98, 1.00] |

## AUROC: removal

| Metric | before (120 images, 60 test) | after (300 images, 150 test) | new test images only (90) |
|---|---|---|---|
| SHAFE-DINO | 0.98 [0.97, 1.00] | 0.98 [0.97, 0.99] | 0.97 [0.96, 0.99] |
| SemRestore (ours) | 0.95 [0.92, 0.98] | 0.96 [0.94, 0.97] | 0.96 [0.94, 0.98] |

## AUROC: substitution

| Metric | before (120 images, 60 test) | after (300 images, 150 test) | new test images only (90) |
|---|---|---|---|
| SHAFE-DINO | 0.99 [0.98, 1.00] | 0.99 [0.97, 0.99] | 0.99 [0.97, 1.00] |
| SemRestore (ours) | 0.97 [0.94, 0.99] | 0.97 [0.95, 0.99] | 0.97 [0.95, 1.00] |

## AUROC: recolor

| Metric | before (120 images, 60 test) | after (300 images, 150 test) | new test images only (90) |
|---|---|---|---|
| SHAFE-DINO | 0.54 [0.44, 0.64] | 0.54 [0.48, 0.59] | 0.54 [0.48, 0.61] |
| SemRestore (ours) | 0.97 [0.95, 0.99] | 0.97 [0.96, 0.98] | 0.98 [0.96, 0.99] |
