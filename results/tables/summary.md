# SemRestore-QA: headline results (held-out test split)

Synthetic edits vs PSNR-matched faithful controls. Higher is better except FNR.

| Metric | AUROC [95% CI] | Win-rate vs blur | Win-rate vs noise | FNR @5% FPR | SBC | Severity ρ (area) | Pointing | Cross-restoration AUROC* |
|---|---|---|---|---|---|---|---|---|
| PSNR | 0.539 [0.527, 0.550] | 0.522 | 0.541 | 0.903 | 0.466 | 0.536 | – | 0.949 |
| SSIM | 0.309 [0.264, 0.343] | 0.230 | 0.031 | 1.000 | 0.205 | 0.915 | – | 0.832 |
| MS-SSIM | 0.428 [0.389, 0.460] | 0.348 | 0.122 | 0.985 | 0.337 | 0.869 | – | 0.949 |
| LPIPS | 0.215 [0.172, 0.250] | 0.019 | 0.087 | 1.000 | 0.068 | 0.850 | – | 0.782 |
| DISTS | 0.366 [0.317, 0.414] | 0.242 | 0.281 | 0.985 | 0.132 | 0.581 | – | 0.858 |
| DINO global | 0.592 [0.539, 0.644] | 0.516 | 0.679 | 0.888 | 0.395 | 0.452 | – | 0.864 |
| SHAFE-ResNet (official) | 0.751 [0.720, 0.784] | 0.839 | 0.949 | 0.791 | 0.637 | 0.346 | 0.969 | 0.981 |
| SHAFE-DINO | 0.662 [0.617, 0.703] | 0.770 | 0.704 | 0.536 | 0.577 | 0.315 | 0.837 | 1.000 |
| SHAFE-DINO + calib | 0.639 [0.593, 0.681] | 0.708 | 0.668 | 0.536 | 0.556 | 0.339 | – | 1.000 |
| SemRestore (ours) | 0.955 [0.937, 0.973] | 0.975 | 1.000 | 0.245 | 0.966 | 0.409 | 1.000 | 1.000 |
| SemRestore top-k | 0.945 [0.925, 0.964] | 0.981 | 1.000 | 0.332 | 0.962 | 0.504 | – | 1.000 |

*Real restorer outputs (edit-then-restore), mean over families; see t6 for each family.
Win-rate 0.5 = coin flip. Matched pairs differ by at most 0.1 dB PSNR, so PSNR's win-rate there is essentially arbitrary.
