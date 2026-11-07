# SemRestore-QA: headline results (held-out test split)

Synthetic edits vs PSNR-matched faithful controls. Higher is better except FNR.

| Metric | AUROC [95% CI] | Win-rate vs blur | Win-rate vs noise | FNR @5% FPR | SBC | Severity ρ (area) | Pointing | Cross-restoration AUROC* |
|---|---|---|---|---|---|---|---|---|
| PSNR | 0.533 [0.526, 0.541] | 0.499 | 0.501 | 0.928 | 0.480 | 0.568 | – | 0.927 |
| SSIM | 0.297 [0.271, 0.321] | 0.174 | 0.046 | 0.998 | 0.183 | 0.865 | – | 0.815 |
| MS-SSIM | 0.414 [0.391, 0.434] | 0.309 | 0.145 | 0.986 | 0.317 | 0.839 | – | 0.927 |
| LPIPS | 0.205 [0.179, 0.229] | 0.023 | 0.089 | 1.000 | 0.057 | 0.862 | – | 0.773 |
| DISTS | 0.348 [0.314, 0.380] | 0.222 | 0.298 | 0.994 | 0.123 | 0.657 | – | 0.859 |
| DINO global | 0.570 [0.535, 0.602] | 0.481 | 0.662 | 0.907 | 0.376 | 0.453 | – | 0.830 |
| SHAFE-ResNet (official) | 0.748 [0.730, 0.767] | 0.824 | 0.938 | 0.774 | 0.645 | 0.239 | 0.919 | 0.979 |
| SHAFE-DINO | 0.668 [0.641, 0.695] | 0.741 | 0.743 | 0.561 | 0.583 | 0.306 | 0.851 | 0.997 |
| SHAFE-DINO + calib | 0.641 [0.612, 0.668] | 0.693 | 0.694 | 0.561 | 0.557 | 0.329 | – | 0.998 |
| SemRestore (ours) | 0.962 [0.952, 0.972] | 0.989 | 1.000 | 0.224 | 0.977 | 0.211 | 1.000 | 1.000 |
| SemRestore top-k | 0.952 [0.942, 0.964] | 0.989 | 1.000 | 0.259 | 0.970 | 0.296 | – | 1.000 |

*Real restorer outputs (edit-then-restore), mean over families; see t6 for each family.
Win-rate 0.5 = coin flip. Matched pairs differ by at most 0.1 dB PSNR, so PSNR's win-rate there is essentially arbitrary.
