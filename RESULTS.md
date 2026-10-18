# SemRestore-QA: results (first full run, 19 Oct 2026)

Run on an 8 GB Apple M2 Mac, about 9 hours end to end. Every number comes from
`results/tables/*.csv` and can be regenerated with `scripts/07_evaluate.py`. All numbers are from
the **held-out test split**; the metric was frozen on the dev split first (commit `5c27959`, before
any test evaluation). 95% confidence intervals come from 1,000 bootstrap resamples of images.

## What we ran

| | Dev (tuning) | Test (held out) |
|---|---|---|
| COCO val2017 images (512×512, one target object each) | 60 | 60 |
| Valid edits: addition / removal / substitution / recolor | 16 / 47 / 30 / 114 | 14 / 51 / 32 / 99 |
| PSNR-matched controls: blur / noise | 173 / 207 | 161 / 196 |
| Restorer outputs (Swin2SR, Real-ESRGAN, SD ×4 upscaler) | – | 3 × 113 |

Edit validity is checked automatically with CLIP (a different model family from every scoring
model). Yield: removal 82%, recolor 59%, substitution 52%, addition 36% (after a second seed pass).

**Limitation found:** substitutions change so much (median PSNR 15 dB) that even extreme blur
cannot reach the same PSNR, so 42 of 62 substitutions have no blur control. Every edit has a
noise control.

## Result 1: PSNR, SSIM, LPIPS and DISTS prefer "sharp but wrong" (HalluGen holds on natural photos)

Paired test: each edit against a faithful control with the **same PSNR** (±0.1 dB).
Win-rate = how often the metric says the faithful control is closer to the reference (0.5 = coin flip).

| Metric | vs blur control | vs noise control |
|---|---|---|
| SSIM | 0.23 [0.13, 0.34] | 0.03 [0.00, 0.07] |
| LPIPS | **0.02** [0.00, 0.04] | 0.09 [0.05, 0.13] |
| DISTS | 0.24 [0.16, 0.32] | 0.28 [0.20, 0.35] |
| SHAFE-ResNet (official recipe) | 0.84 [0.77, 0.91] | 0.95 [0.91, 0.98] |
| SHAFE-DINO | 0.77 [0.67, 0.86] | 0.70 [0.63, 0.78] |
| **SemRestore (ours)** | **0.98** [0.95, 0.99] | **1.00** [1.00, 1.00] |

LPIPS picks the hallucinated image in 98% of blur pairs. Figure: `results/figures/fig2_winrate.png`.

## Result 2: detection AUROC per edit type (edits vs matched controls)

| Metric | Addition | Removal | Substitution | Recolor | F1 | FNR @5% FPR ↓ |
|---|---|---|---|---|---|---|
| SSIM | 0.18 | 0.11 | 0.11 | 0.29 | 0.52 | 1.00 |
| LPIPS | 0.01 | 0.05 | 0.02 | 0.23 | 0.53 | 1.00 |
| DISTS | 0.09 | 0.20 | 0.17 | 0.45 | 0.53 | 0.98 |
| SHAFE-ResNet | 0.93 | 0.89 | 0.91 | 0.77 | 0.61 | 0.79 |
| SHAFE-DINO | **1.00** | **0.98** | **0.99** | 0.54 | 0.60 | 0.54 |
| **SemRestore (ours)** | 0.96 | 0.95 | 0.97 | **0.97** | **0.85** | **0.24** |

Read honestly: SHAFE-DINO is slightly better than ours on object edits (by about 0.03), but it is
at chance on recolouring. **DINOv2 is trained with colour-jitter augmentation, so it is close to
colour-blind.** Our colour map fixes that, and SemRestore is the only metric that is good on all four
types. It misses 24% of edits at a 5% false-alarm rate; SHAFE-DINO misses 54%.

## Result 3: ablation (what each component adds)

| Variant | Dev mean AUROC (per type) | Test AUROC (all edits) | Win-rate blur / noise |
|---|---|---|---|
| SHAFE-DINO | 0.873 | 0.662 | 0.77 / 0.70 |
| + calibration | 0.857 | 0.639 | 0.71 / 0.67 |
| + calibration + colour map = **SemRestore** | **0.978** | **0.955** | 0.98 / 1.00 |
| SemRestore with top-k pooling | 0.973 | 0.945 | 0.98 / 1.00 |

* Calibration alone does **not** help. Its job is to put the DINO maps and the colour map on one
  scale ("how unusual compared with blur/noise"), so the two can be added.
* Top-k pooling (in our original plan) did **not** beat SHAFE's softmax pooling, so we dropped it.
* We added the colour map *after* seeing on dev that SHAFE-DINO fails on recolor. That decision used
  dev only; the test numbers above were computed once, after the freeze.

## Result 4: sharpness bias and localization

* Sharpness-bias curve (clean reference blurred more and more vs the edit; area, higher is better):
  LPIPS 0.07, DISTS 0.13, SSIM 0.20, PSNR 0.47, SHAFE-DINO 0.58, SHAFE-ResNet 0.64,
  **SemRestore 0.97**. Figure: `fig3_sharpness_bias.png`.
* Heatmaps (does the hottest point fall inside the edited region?): pointing game SemRestore 1.00,
  SHAFE-ResNet 0.97, SHAFE-DINO 0.84 (0.68 on recolor). IoU 0.84 / 0.79 / 0.79. Figure: `fig1_examples.png`.

## Result 5: cross-restoration (edit-then-restore, real restorer outputs, reported separately)

We super-resolve the edited image with each restorer and compare to the clean reference.

| Family | SSIM | LPIPS | PSNR | SHAFE-ResNet | SHAFE-DINO | SemRestore |
|---|---|---|---|---|---|---|
| Regression (Swin2SR) | 0.89 | 0.75 | 0.98 | 0.98 | 1.00 | 1.00 |
| GAN (Real-ESRGAN) | 0.80 | 0.82 | 0.93 | 0.96 | 1.00 | 1.00 |
| Diffusion (SD ×4) | 0.80 | 0.78 | 0.94 | 1.00 | 1.00 | 1.00 |

The localized metrics keep working on real restorer outputs from all three families. **Caveat:**
this test is not PSNR-matched, so it is easy (even PSNR gets 0.93–0.98). The human study's
"real vs real" pairs are the harder test of real restorers.

## Where our metric is weak (say this in the talk)

* **Severity.** Spearman between score and edit size (area) is only 0.28–0.52 for SemRestore;
  pixel metrics track area better (SSIM 0.78–0.98), because a bigger edit simply changes more
  pixels. For recolor severity (30°/90°/150° hue shift): SemRestore 0.61, SHAFE-ResNet 0.72, SSIM 0.71.
  Our metric answers "did content change?" well, and "how much?" less well.
* **Small test set** for addition (14 test edits): wide intervals.
* SHAFE-DINO beats us slightly on object edits; the colour map costs about 0.03 there.

## SHAFE re-implementation check

`scripts/check_shafe_official.py` runs the official HalluGen `SHAFE.py` and ours on 20 real
image pairs: maximum difference **0.0** (identical). One fix was needed: the official code resizes
to 224×224 before the ResNet; the first draft did not.

## Not done yet

* **Human study:** `results/human/rating.html` is built (121 pairs: 41 edit-vs-blur, 50 real-vs-real,
  20 severity, 10 gold; 35 MB, works offline). It needs 5–8 raters, then `scripts/09_analyze_human.py`.
* Deblurring as a second restoration task (optional in the plan).
