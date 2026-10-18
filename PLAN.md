# SemRestore-QA: Project Plan

**Semantic-fidelity evaluation for image restoration**
CV course project · 3 people · plan dated 18 Oct 2026

---

## 1. The problem in one paragraph

Image restoration (super-resolution, deblurring) has to guess detail that the degraded input does
not contain. Generative restorers guess *confidently*: the output looks sharp, but it may contain
an object that was never there, miss one that was, or change what something is. The usual quality
metrics do not catch this. PSNR and SSIM compare pixels, and LPIPS averages feature differences
over the whole image, so a small but important change gets diluted. HalluGen (CVPR 2026) showed
that PSNR, SSIM and LPIPS can score a hallucinated image *higher* than a slightly blurred but
correct one.

**Our question:** can we measure *content* faithfulness on natural photos, for several kinds of
content change, and does a simple localized metric do it better than the existing ones?

## 2. How the brief maps to what we do

| Brief asks for | What we do |
|---|---|
| Metric that penalizes semantic changes missed by PSNR/SSIM/LPIPS | A localized metric on DINOv2 patch features that outputs a score **and** a heatmap |
| Validated by controlled hallucinations | 4 edit types with known mask, type and severity, compared against PSNR-matched faithful controls |
| …human ratings | Blinded pairwise study, ~130 pairs, 5–8 raters, held-out images only |
| …cross-restoration tasks | ×4 super-resolution with 3 restorer families (regression, GAN, diffusion) |
| Spearman ρ, Pearson r | Metric vs human vote share; metric vs edit severity |
| Hallucination AUROC / F1 | Edits vs matched controls, F1 at a threshold fixed on dev |
| Severity correlation | Spearman between metric and edit size / colour-shift amount |
| False-negative rate ↓ | FNR at the threshold that gives 5% false positives on dev |
| Perturbations only from train/val images | All source images are COCO train/val images (never COCO test). Within them, a held-out split is never used for tuning |
| Blinded pairwise ratings on a held-out subset | Human study uses only held-out test images; A/B side and order randomised per rater |
| Multiple restoration families | Swin2SR (regression), Real-ESRGAN (GAN), SD ×4 upscaler (diffusion) |
| Risk: human-study design quality | Fixed question + instructions, gold checks, exclusion rule set before collection, agreement reported with a human ceiling |

## 3. Background (what the class needs to know)

**Restoration is ill-posed.** We observe y = A(x) + n (A = downsample or blur, n = noise). Many
sharp images x fit the same y, so the restorer must guess. The *perception–distortion trade-off*
says sharper-looking guesses tend to be further from the truth.

**Two kinds of hallucination (HalluGen's terms).** *Intrinsic*: the output contradicts the input
(A(x̂) ≠ y). *Extrinsic*: the output is consistent with the input but still wrong. Extrinsic ones
can only be found by comparing with the reference, which is what full-reference metrics do.

**Why current metrics miss it.** PSNR/SSIM look at pixels; a blurry image loses many pixels a
little, a hallucination loses few pixels a lot, and the average can come out the same.
LPIPS/DISTS use deep features but average over the image, so one wrong object is diluted.

**What HalluGen / SHAFE did.**
- Generated controllable hallucinations in brain MRI with a diffusion model (4,350 images).
- Proposed **SHAFE**: low-pass both images, compare deep features patch by patch (cosine
  distance), then pool with a softmax that puts most weight on the worst patches:
  SHAFE = Σᵢ wᵢ dᵢ with wᵢ = softmax(dᵢ / τ).
- SHAFE beat PSNR/SSIM/LPIPS clearly (e.g. real-output AUC 0.79 for SHAFE-DINOv3 vs 0.42 for LPIPS).
- Stated limits: MRI only (plus small demos), no explicit semantic control, and manual patch
  labels had low agreement between two experts (Cohen's κ = 0.30).

## 4. What is new in our project (and what is not)

We do **not** claim the first hallucination benchmark or the first localized metric. HalluGen
already has both. Our contribution is a careful test of the idea in a setting they did not cover:

| HalluGen / SHAFE | SemRestore-QA |
|---|---|
| Mostly brain MRI | Natural photos (COCO) |
| Hallucinations defined by the forward operator | Semantic edit types: **addition, removal, substitution, recolor** |
| Severity = pixel error | Severity = edit size, and hue-shift amount for recolor (3 levels on the same object) |
| Blur comparison as a figure | **PSNR-matched controls** for every edit: PSNR literally cannot separate them |
| SHAFE is the method | SHAFE is the **baseline we must beat** (re-implemented from the official code) |
| 300 real outputs with binary expert labels | **Edit-then-restore**: real restorer outputs with a known content change, 3 families |
| Patch labels from 2 experts (κ = 0.30) | Pairwise "which is closer?" judgments, which are easier and more reliable than drawing regions |

## 5. Method

### 5.1 Data
- 300 COCO images, each center-cropped to 512×512, each with one target object covering 3–25%
  of the image.
- **Dev split (150):** used for calibration, tuning and thresholds.
- **Test split (150):** held out. Final numbers and the human study come only from here.

### 5.2 Controlled edits (the ground truth)

| Type | How | Model |
|---|---|---|
| Addition (a hallucinated object) | Inpaint a new object into an empty background square | Stable Diffusion inpainting |
| Removal | Erase the object, fill with background | LaMa |
| Substitution | Replace the object with a different category | Stable Diffusion inpainting |
| Recolor (attribute change) | Rotate the object's hue by 30°, 90°, 150° | none (HSV shift) |

Two rules make the labels trustworthy:
1. **Paste-back.** Everything outside the edit mask is copied exactly from the original, so the
   mask is exactly where content changed (unit-tested).
2. **Validity check.** A CLIP zero-shot check on the edited region (e.g. for substitution: does
   the region now look more like "a dog" than "a chair"?). Failed edits are dropped. CLIP is a
   different model family from our metric encoders, so it cannot bias the metric.

### 5.3 Faithful controls: the key test
For every edit with PSNR = p dB, we blur the clean reference until its PSNR is also p (±0.1 dB),
and separately add noise until it is p. Now we have two images that PSNR cannot tell apart:
one *sharp but wrong*, one *blurry but right*. **A good metric must prefer the blurry one.**

### 5.4 Real restoration (cross-family)
We downsample test images ×4 (512 → 128) and restore them with three families:
Swin2SR (regression), Real-ESRGAN (GAN), SD ×4 upscaler (diffusion).

**Edit-then-restore.** We also restore the *edited* images and compare their output with the
*clean* reference. That output is exactly a restorer output containing a content change, with
realistic restorer artifacts on both sides. So we get an AUROC per family on real restorer outputs
with known labels. The plain restorations of clean images go into the human study, where humans
are the only ground truth.

### 5.5 Metrics compared

| Group | Metrics |
|---|---|
| Pixel | PSNR, SSIM, MS-SSIM |
| Deep feature, global | LPIPS, DISTS, DINOv2 CLS cosine |
| Localized (baseline) | **SHAFE-ResNet**, same recipe as official `SHAFE.py` (ResNet-50 features at 2 layers, FFT low-pass, softmax τ = 0.02) |
| Localized (baseline) | **SHAFE-DINO**: the SHAFE recipe on DINOv2 features |
| Ours | **SemRestore** = SHAFE-DINO + colour map + calibration (softmax pooling kept) |

**Our metric in four lines** (for patch p and DINOv2 layer l):

```
d_l(p) = 1 − cos( feat_l(ref)[p], feat_l(test)[p] )          patch distance (images low-passed first)
c(p)   = | mean Lab colour of patch p (ref) − (test) |        colour map (a*, b* only)
z(p)   = max(0, (d(p) − μ) / σ)  for every layer and for c    calibration: μ, σ = typical value
                                                             caused by blur/noise alone on DEV images
Z(p)   = mean of the z maps  → heatmap;   score = Σ softmax(Z/τ)·Z   (SHAFE's pooling)
```

The two ideas are simple to explain:
- **Colour map:** DINOv2 is trained with colour-jitter augmentation, so it is close to colour-blind
  (SHAFE-DINO is at chance on recoloured objects). A per-patch colour difference fixes that.
- **Calibration:** "how unusual is this patch's difference, compared with what blur or noise alone
  would cause?" It puts the DINO maps and the colour map on one scale so they can be averaged.

**What changed after the first run (dev data only).** Our original plan used top-k pooling instead
of the colour map. On dev, top-k and calibration alone did not beat SHAFE-DINO, and the per-type
table showed the recolor failure. We added the colour map, kept SHAFE's softmax pooling, froze on dev,
and only then evaluated on test. Results: [RESULTS.md](RESULTS.md).

**Ablation.** Every step is reported separately: SHAFE-DINO → + calibration → + colour map
(= SemRestore), plus a top-k pooling variant.

### 5.6 No tuning on test (the freeze)
Calibration statistics, hyperparameters (which layers, τ, k) and all thresholds are chosen on the
dev split only. Script 06 then writes `freeze.json` with a fingerprint of the metric code, the
config and the split. Scripts 07, 08 and 10 refuse to touch test results if anything changed.

## 6. Evaluation: what each number answers

| Number | Question it answers | Table |
|---|---|---|
| AUROC [95% CI] | Over all edit/control pairs, how often does the metric rank the edit as worse? | t1 |
| F1 at the dev threshold | If we use the metric as a yes/no detector, how good is it? | t1 |
| FNR at 5% FPR | Allowing 5% false alarms on faithful images, how many real changes slip through? | t1 |
| Paired win-rate | For each edit and its own same-PSNR control: does the metric prefer the control? (0.5 = coin flip) | t2 |
| Sharpness-bias curve | As we blur the clean image more, when does the metric start preferring the hallucination? | t3 |
| Severity Spearman / Pearson | Bigger edit → higher score? | t4 |
| Pointing game, IoU, pixel AUROC | Does the heatmap point at the edited region? | t5 |
| Edit-then-restore AUROC per family | Does it still work on real restorer outputs? | t6 |
| Ablation | What does each component add? | t7 |
| Metric vs human: accuracy, ρ, r | Does the metric agree with people? | h3 |

Confidence intervals come from 1,000 bootstrap resamples **of images** (several items come from the
same image, so they are not independent). Synthetic-edit and real-restorer results are always in
separate tables, never averaged together.

## 7. Human study

**Format.** Raters see the reference on top and two candidates A and B below, and answer
*"Which image preserves the CONTENT of the reference better?"* with A / B / About the same.
The instructions say explicitly to ignore blur, noise and prettiness.

**Pairs (~130, about 15–20 minutes per rater), held-out test images only:**

| Type | n | What | Why |
|---|---|---|---|
| Edit vs blur control | 50 | sharp-but-wrong vs blurry-but-right, same PSNR | Do humans see the change? Does the metric agree with them? |
| Real vs real | 50 | two restorer families on the same image | Real outputs, where humans are the only ground truth |
| Severity | 20 | small (30°) vs large (150°) recolor of the same object | Human severity ordering |
| Gold | 10 | exact copy of the reference vs an obvious edit | Rater attention check |

**Running it.** Script 08 writes one `rating.html` file with the images embedded. Raters open it
in any browser, type a nickname, rate, and the page downloads a CSV for them to send back. There
is no server and no personal data. A/B side and order are shuffled per rater, and the same source
image never appears twice in a row.

**Rules fixed before collection.** Raters with under 80% on gold pairs are excluded. Raw CSVs
are never edited. We report Krippendorff's α and split-half agreement (the human ceiling) next to
the metric's agreement with the human majority.

**Honest limits.** 5–8 classmates is a convenience sample and ~130 pairs gives wide intervals
(about ±9 points overall, wider within each pair type), so we call it a **pilot** and always show the intervals.

## 8. Expected results, and what we do if they do not happen

**What we expect.** PSNR is a coin flip on matched pairs by construction. SSIM and LPIPS
should *prefer the edit* (win-rate below 0.5) against blur controls. SHAFE should do much better,
and our metric should be at least as good, especially on removal and substitution and on blur
controls.

**If our metric does not beat SHAFE:** we report it honestly. The contribution is then the
natural-image test set with matched controls, the edit-then-restore protocol, the human pilot, and
an analysis of which content changes SHAFE still misses. That is still a complete, useful project.

| Risk | Plan |
|---|---|
| Edits look fake or fail | CLIP filter plus a look at `edit_samples.png` before the full run |
| Colab time limits | Every script resumes; reduce `n_images` in `config.yaml` if needed |
| Diffusion upscaler too slow | Use fewer images for that family (`restore.n_images`) |
| Few raters | Report the real number and the intervals; do not over-claim |
| A model ID disappears | Change it in `config.yaml`, mention it in the report |

## 9. Team split and schedule

| Person | Owns | Scripts |
|---|---|---|
| A: Data | images, edits, controls, restorations | 01–04 |
| B: Metric | baselines, SHAFE check, our metric, tuning, ablation | 05–07 |
| C: Evaluation and human study | human study, figures, report integration | 08–10 |

| Dates | Work | Output |
|---|---|---|
| **Sat 18 – Wed 22 Oct** | Run tests and smoke test. Run 01–03 on ~30 images, 05 on them, first baseline numbers. Recruit raters | **Midterm**: real example images + first win-rate numbers |
| Thu 23 – Wed 29 Oct | Full 300 images, 04 restorations, 05 scoring, 06 freeze, 07 evaluation | All synthetic and cross-restoration tables |
| Thu 30 Oct – Wed 5 Nov | 08 human study (one calendar week of rating), 09 analysis, 10 figures, failure analysis | Human tables, figures, report, final slides |

## 10. Midterm slides (about 10 slides)

1. **Title** – team, guide, date.
2. **Problem** – one picture: reference / sharp-but-wrong / blurry-but-right, with their PSNR,
   SSIM, LPIPS values. Ask the class which one is "better".
3. **Why metrics fail** – pixel averaging vs one wrong object; perception–distortion trade-off.
4. **Prior work** – HalluGen and SHAFE (the formula), and their limits (MRI only, κ = 0.30).
5. **Our question and what is new** – the table in §4.
6. **Benchmark** – the 4 edit types with examples (`edit_samples.png`), and the paste-back rule.
7. **Key test** – PSNR-matched controls, explained in one sentence.
8. **Our metric** – the four-line formula and the two ideas (colour map, calibration).
9. **Cross-family and human study** – edit-then-restore diagram; the rating-page screenshot.
10. **Evaluation and protocol** – metrics table (§6) and the dev/test freeze.
11. **Progress so far** – only what actually ran: N images, example edits, first win-rates.
12. **Plan and risks** – schedule and the "if it does not beat SHAFE" answer.

A real image and one real number are worth more than a long list of plans.

## 11. What we simplified compared with the earlier drafts

- One `config.yaml` and ten numbered scripts instead of many configs and make targets.
- Two splits (dev / test) instead of three. Calibration uses only *degraded clean* dev images,
  never edits.
- Severity comes from the edit itself (size, hue shift), so no extra segmentation model is needed.
- Attribute change kept for free via recolor (no generative model).
- Removal uses LaMa, which removes cleanly instead of painting something new.
- Real restorers get ground truth through edit-then-restore instead of proxy labels.
- Human study is a single offline HTML file instead of a web server with tokens.
- One restoration task (×4 SR) with three families. Deblurring is optional future work.
- Dropped: saliency weighting, Steiger's test, VLM judge.

## 12. Questions to confirm with the instructor

1. Is our reading of "perturbations only from training/validation images" OK? All images come from
   COCO train/val, and our held-out split is never used for tuning. If they mean our *own* train
   split only, we keep edit-vs-blur pairs out of the human study and use only real-vs-real pairs there.
2. Is one task with three restorer families enough for "cross-restoration", or is deblurring required?
3. Does the human study need a consent or ethics note beyond the consent text on the rating page?
