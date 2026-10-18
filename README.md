# SemRestore-QA

**Does an image-quality metric notice when a restoration model changes *what* is in the image?**

Generative restorers (GANs, diffusion) make images look sharp, but sometimes they add, remove or
change objects. PSNR, SSIM and LPIPS often score such a "sharp but wrong" output *higher* than a
slightly blurry but correct one (HalluGen, CVPR 2026, showed this on brain MRI). We test this on
natural photos, with four kinds of content change, and build a simple metric that catches them.

The project plan (motivation, method, evaluation, human study, schedule, slide outline) is in
**[PLAN.md](PLAN.md)**. This README is about running the code.

---

## The idea in one picture

```
 COCO photo ──► controlled edit (known mask, type, severity) ──► "sharp but WRONG"
     │                                                                │
     └────────► blur / noise to the SAME PSNR ───────────────────► "blurry but RIGHT"
                                                                      │
                     every metric scores both  ──►  does it prefer the right one?
```

Because the two images have the same PSNR, a metric can only pass this test if it actually looks
at content. On top of that we run three real super-resolution models (regression, GAN, diffusion)
and a small blinded human study.

## Quick start

On Colab (recommended, free T4 GPU): open `notebooks/colab_run_all.ipynb` and run the cells.

Locally:

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # 19 fast tests, CPU, no downloads
python scripts/00_smoke_test.py           # whole pipeline on synthetic images, ~8 min CPU
```

The smoke test uses random-weight networks, so its numbers mean nothing. It only proves that every
step, table, figure and the human-study page work.

## Run order

Everything reads `config.yaml`. Each script skips work that is already done, so re-running after
a Colab disconnect is safe.

| Script | Does | Main output |
|---|---|---|
| `01_build_dataset.py` | 300 COCO images, one editable object each, 150 dev / 150 test | `data/images.csv` |
| `02_make_edits.py` | addition, removal, substitution, recolor (×3 severities) + CLIP validity check | `data/edits/edits.csv`, `results/figures/edit_samples.png` |
| `03_make_controls.py` | blur- and noise-matched controls (same PSNR as each edit) + fixed degradation grid | `data/controls/`, `data/degraded/` |
| `04_restore.py` | ×4 SR with Swin2SR, Real-ESRGAN, SD ×4 upscaler on test images (clean and edited) | `data/restored/restored.csv` |
| `05_score.py` | all metrics on all pairs; saves SHAFE and DINOv2 patch-distance maps | `results/scores.csv`, `results/maps/` |
| `06_freeze.py` | calibrate + tune **on dev only**, set thresholds, write the freeze file | `results/freeze.json` |
| `07_evaluate.py` | final numbers **on held-out test** | `results/tables/t1…t7`, `summary.md` |
| `08_human_pairs.py` | builds a single offline `rating.html` from test images | `results/human/` |
| `09_analyze_human.py` | rater QC, agreement, metric vs human | `results/tables/h1…h4` |
| `10_figures.py` | figures for slides and report | `results/figures/fig1…fig5` |

`07`, `08` and `10` refuse to run if `sqa/metrics.py`, the metric/tuning config, or the dev/test
split changed after `06` froze the metric. That is how we guarantee nothing was tuned on test.

## Repository map

```
config.yaml              every setting in one place
sqa/                     the library (short, commented files)
  data.py                COCO selection, crop, masks, split
  edits.py               SD inpainting, LaMa, recolor, CLIP check
  controls.py            PSNR matching by bisection
  restorers.py           Swin2SR / Real-ESRGAN / SD x4 upscaler
  metrics.py             PSNR…DISTS, SHAFE-ResNet (official recipe), DINOv2 maps, our metric
  evaluation.py          AUROC, F1, FNR@FPR, win-rate, SBC, bootstrap CI, localization, Krippendorff
  items.py               the table of everything that gets scored
  scoring.py             badness table, freeze fingerprint
  human.py               pair building, the rating page, rating analysis
  viz.py                 plotting helpers
scripts/00…10            the pipeline (see table above)
tests/test_core.py       unit tests for every rule we rely on
notebooks/               Colab notebook
```

## Models used (all public, verify they download on your machine)

| Role | Model | Family |
|---|---|---|
| Addition / substitution | `stable-diffusion-v1-5/stable-diffusion-inpainting` | Stable Diffusion |
| Removal | Big-LaMa (TorchScript, GitHub release) | LaMa |
| Edit validity check | `openai/clip-vit-base-patch32` | OpenAI CLIP |
| Our metric, SHAFE-DINO | `facebook/dinov2-base` | DINOv2 |
| SHAFE-ResNet | `timm: resnetaa50d.d_in12k` (same as official SHAFE.py) | ResNet / ImageNet |
| SR regression | `caidas/swin2SR-classical-sr-x4-64` | |
| SR GAN | Real-ESRGAN x4plus | |
| SR diffusion | `stabilityai/stable-diffusion-x4-upscaler` | |

The models that **make or check** edits never share a family with the models that **score**
them (tested in `test_encoder_families_are_separate`). If a model ID has moved on Hugging Face,
change it in `config.yaml` and note it in your report.

## Data and licences

COCO images are under Flickr licences and the annotations are CC BY 4.0. Do not commit images to
git (`data/` is ignored). Check each model's licence before releasing edited images.
