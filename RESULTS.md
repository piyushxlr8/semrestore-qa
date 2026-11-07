# SemRestore-QA: results (300 images, 8 Nov 2026)

Run on an 8 GB Apple M2 Mac. Every number comes from `results/tables/*.csv` and can be regenerated
with `scripts/07_evaluate.py` and `scripts/11_compare_scaling.py`. All numbers are from the
**held-out test split**. The metric was frozen on the dev split first (commit `68dc04c`) before any
test evaluation. 95% confidence intervals come from 1,000 bootstrap resamples of images.

The first run used 120 images (results kept in `results/snapshot_120/`, git tag `results-120`).
After the professor asked for more data we added 180 new COCO images without touching the original
120. The before/after comparison is at the end of this file; **no conclusion changed.**

## What we ran

| | Dev (tuning) | Test (held out) |
|---|---|---|
| COCO val2017 images (512×512, one target object each) | 150 (60 original + 90 new) | 150 (60 original + 90 new) |
| Valid edits: addition / removal / substitution / recolor | 45 / 123 / 82 / 288 | 49 / 124 / 74 / 270 |
| PSNR-matched controls: blur / noise | 452 / 538 | 437 / 517 |
| Restorer outputs (Swin2SR, Real-ESRGAN, SD ×4 upscaler) | – | 3 × 288 |

Edit validity is checked automatically with CLIP (a different model family from every scoring
model). Yield over all 300 images: removal 82%, recolor 62%, substitution 52%, addition 45% (after
one retry pass with new seeds).

**Limitation:** substitutions change so much (median PSNR 16 dB) that even extreme blur cannot reach
the same PSNR, so 98 of 156 substitutions have no blur control. Every edit has a noise control.

## Result 1: PSNR, SSIM, LPIPS and DISTS prefer "sharp but wrong"

Paired test: each edit against a faithful control with the **same PSNR** (±0.1 dB).
Win-rate = how often the metric says the faithful control is closer to the reference (0.5 = coin flip).

| Metric | vs blur control (437 pairs) | vs noise control (517 pairs) |
|---|---|---|
| SSIM | 0.17 [0.12, 0.23] | 0.05 [0.02, 0.07] |
| LPIPS | **0.02** [0.01, 0.04] | 0.09 [0.06, 0.12] |
| DISTS | 0.22 [0.17, 0.27] | 0.30 [0.25, 0.35] |
| SHAFE-ResNet (official recipe) | 0.82 [0.78, 0.87] | 0.94 [0.91, 0.96] |
| SHAFE-DINO | 0.74 [0.69, 0.80] | 0.74 [0.69, 0.79] |
| **SemRestore (ours)** | **0.99** [0.98, 1.00] | **1.00** [1.00, 1.00] |

LPIPS picks the hallucinated image in 98% of blur pairs. Figure: `results/figures/fig2_winrate.png`.

## Result 2: detection AUROC per edit type (edits vs matched controls)

| Metric | Addition (49) | Removal (124) | Substitution (74) | Recolor (270) | F1 | FNR @5% FPR ↓ |
|---|---|---|---|---|---|---|
| SSIM | 0.31 | 0.09 | 0.10 | 0.27 | 0.52 | 1.00 |
| LPIPS | 0.07 | 0.03 | 0.02 | 0.23 | 0.52 | 1.00 |
| DISTS | 0.13 | 0.15 | 0.13 | 0.44 | 0.53 | 0.99 |
| SHAFE-ResNet | 0.95 | 0.88 | 0.89 | 0.74 | 0.61 | 0.77 |
| SHAFE-DINO | **0.99** | **0.98** | **0.99** | 0.54 | 0.57 | 0.56 |
| **SemRestore (ours)** | **0.99** | 0.96 | 0.97 | **0.97** | **0.87** | **0.22** |

SHAFE-DINO is near-perfect on object edits but at chance on recolouring: **DINOv2 is trained with
colour-jitter augmentation, so it is close to colour-blind.** Our colour map fixes that. SemRestore
is the only metric that is good on all four types; it ties SHAFE-DINO on additions and trails it by
0.02 on removal and substitution. At a 5% false-alarm rate it misses 22% of edits; SHAFE-DINO
misses 56%.

## Result 3: ablation (what each component adds)

| Variant | Dev mean AUROC (per type) | Test AUROC (all edits) | Win-rate blur / noise |
|---|---|---|---|
| SHAFE-DINO | 0.861 | 0.668 | 0.74 / 0.74 |
| + calibration | 0.848 | 0.641 | 0.69 / 0.69 |
| + calibration + colour map = **SemRestore** | **0.973** | **0.962** | 0.99 / 1.00 |
| SemRestore with top-k pooling | 0.968 | 0.952 | 0.99 / 1.00 |

* Calibration alone does **not** help. Its job is to put the DINO maps and the colour map on one
  scale ("how unusual compared with blur/noise"), so the two can be added.
* Top-k pooling (in our original plan) did **not** beat SHAFE's softmax pooling.
* The colour map was added in the 120-image run after seeing, on dev only, that SHAFE-DINO fails on
  recolor. Re-tuning on the 150-image dev split picked exactly the same settings.

## Result 4: sharpness bias and localization

* Sharpness-bias curve (clean reference blurred more and more vs the edit; area, higher is better):
  LPIPS 0.06, DISTS 0.12, SSIM 0.18, PSNR 0.48, SHAFE-DINO 0.58, SHAFE-ResNet 0.64,
  **SemRestore 0.98**. Figure: `fig3_sharpness_bias.png`.
* Heatmaps (does the hottest point fall inside the edited region?): pointing game SemRestore 1.00,
  SHAFE-ResNet 0.92, SHAFE-DINO 0.85 (0.72 on recolor). IoU 0.85 / 0.78 / 0.79. Figure: `fig1_examples.png`.

## Result 5: cross-restoration (edit-then-restore, real restorer outputs, reported separately)

We super-resolve the edited image with each restorer and compare to the clean reference
(138 restored edits vs 150 restored clean images per family).

| Family | SSIM | LPIPS | PSNR | SHAFE-ResNet | SHAFE-DINO | SemRestore |
|---|---|---|---|---|---|---|
| Regression (Swin2SR) | 0.88 | 0.74 | 0.96 | 0.99 | 1.00 | 1.00 |
| GAN (Real-ESRGAN) | 0.80 | 0.81 | 0.91 | 0.96 | 1.00 | 1.00 |
| Diffusion (SD ×4) | 0.77 | 0.77 | 0.91 | 0.98 | 1.00 | 1.00 |

The localized metrics keep working on real restorer outputs from all three families. **Caveat:**
this test is not PSNR-matched, so it is easy (even PSNR gets 0.91–0.96). The human study's
"real vs real" pairs are the harder test of real restorers.

## Where our metric is weak (say this in the talk)

* **Severity.** SemRestore says *whether* content changed much better than *how much*. Spearman
  between its score and edit size: addition 0.21 [−0.08, 0.43], removal 0.15 [−0.02, 0.33],
  substitution 0.28 [0.04, 0.48]; recolor severity (30°/90°/150° hue shift) 0.61 [0.57, 0.66].
  Pixel metrics track size better (SSIM 0.65–0.98), because a bigger edit simply changes more
  pixels. With more data this weakness is **clearer** than in the 120-image run (see below).
* SHAFE-DINO is slightly better on removal and substitution (by 0.02 AUROC).

## SHAFE re-implementation check

`scripts/check_shafe_official.py` runs the official HalluGen `SHAFE.py` and ours on 20 real
image pairs: maximum difference **0.0** (identical).

## Human study (ready, not yet run)

`results/human/rating.html` is rebuilt from the 150-image test split: 130 pairs (50 edit-vs-blur,
50 real-vs-real, 20 severity, 10 gold), 44 MB, works offline. It needs 5–8 raters, then
`scripts/09_analyze_human.py`.

---

## Before/after: 120 vs 300 images

Held-out test split, 95% bootstrap CIs over images. "New only" = just the 90 brand-new test images,
which the first run never saw. Full tables: `results/tables/scaling_summary.md`,
`s1_before_after.csv`, `s2_new_test_only.csv`.

**Win-rate (faithful control judged closer), blur / noise**

| Metric | Before (60 test) | After (150 test) | New only (90) |
|---|---|---|---|
| SSIM | 0.23 [0.13, 0.34] / 0.03 [0.00, 0.07] | 0.17 [0.12, 0.23] / 0.05 [0.02, 0.07] | 0.14 [0.09, 0.20] / 0.06 [0.02, 0.10] |
| LPIPS | 0.02 [0.00, 0.04] / 0.09 [0.04, 0.13] | 0.02 [0.01, 0.04] / 0.09 [0.06, 0.12] | 0.03 [0.00, 0.05] / 0.09 [0.05, 0.14] |
| DISTS | 0.24 [0.16, 0.32] / 0.28 [0.20, 0.35] | 0.22 [0.17, 0.27] / 0.30 [0.25, 0.35] | 0.21 [0.15, 0.27] / 0.31 [0.24, 0.37] |
| SHAFE-ResNet | 0.84 [0.77, 0.91] / 0.95 [0.91, 0.98] | 0.82 [0.78, 0.87] / 0.94 [0.91, 0.96] | 0.82 [0.75, 0.87] / 0.93 [0.89, 0.97] |
| SHAFE-DINO | 0.77 [0.67, 0.86] / 0.70 [0.63, 0.78] | 0.74 [0.69, 0.80] / 0.74 [0.69, 0.79] | 0.72 [0.66, 0.79] / 0.77 [0.70, 0.83] |
| SemRestore | 0.98 [0.95, 0.99] / 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] / 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] / 1.00 [1.00, 1.00] |

**AUROC per edit type**

| Edit type | SHAFE-DINO before → after (new only) | SemRestore before → after (new only) |
|---|---|---|
| Addition | 1.00 [0.98, 1.00] → 0.99 [0.98, 1.00] (0.99) | 0.96 [0.91, 1.00] → **0.99** [0.97, 1.00] (0.99) |
| Removal | 0.98 [0.97, 1.00] → 0.98 [0.97, 0.99] (0.97) | 0.95 [0.92, 0.98] → 0.96 [0.94, 0.97] (0.96) |
| Substitution | 0.99 [0.98, 1.00] → 0.99 [0.97, 0.99] (0.99) | 0.97 [0.94, 0.99] → 0.97 [0.95, 0.99] (0.97) |
| Recolor | 0.54 [0.44, 0.64] → 0.54 [0.48, 0.59] (0.54) | 0.97 [0.95, 0.99] → 0.97 [0.96, 0.98] (0.98) |

**FNR at the dev-frozen 5% FPR threshold (lower is better)**

| Metric | Before | After | New only |
|---|---|---|---|
| SHAFE-ResNet | 0.79 [0.72, 0.85] | 0.77 [0.73, 0.81] | 0.77 [0.72, 0.82] |
| SHAFE-DINO | 0.54 [0.47, 0.60] | 0.56 [0.53, 0.60] | 0.58 [0.53, 0.62] |
| SemRestore | 0.24 [0.18, 0.31] | 0.22 [0.19, 0.27] | 0.22 [0.17, 0.27] |

**Did any conclusion change? No.**

* All main findings hold, with intervals roughly half as wide: SSIM/LPIPS/DISTS still prefer the
  hallucinated image; SHAFE-DINO is still at chance on recolor; SemRestore is still the only metric
  good on all four types and still misses far fewer edits.
* The 90 new test images, which played no part in designing the metric, give the same numbers.
* **One gap closed:** on additions SemRestore went from 0.96 to 0.99 and now ties SHAFE-DINO.
  The earlier gap came from having only 14 test additions (now 49).
* **One weakness became clearer:** SemRestore's correlation with edit size fell (addition
  0.52 → 0.21, removal 0.28 → 0.15, substitution 0.43 → 0.28), and the intervals for addition and
  removal now include zero. It is a detector of content change, not a measure of its size.
* Dev tuning picked identical settings on 150 dev images as on 60.
