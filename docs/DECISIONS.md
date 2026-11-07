# Decisions log

Every choice that affects the data, the metric or the evaluation is recorded here, newest last.

## Up to the 120-image run (19 Oct 2026)

1. **SHAFE baseline** re-implemented from the official HalluGen `SHAFE.py`, including its 224×224
   resize. Checked against the official code on 20 real pairs: identical scores (difference 0.0).
2. **Dataset size 120** (60 dev / 60 test) so the first run fit one night on an 8 GB M2 Mac.
3. **Edit validity**: removal is valid if CLIP becomes ≥5× less sure the object is there
   (odds ratio); substitution inpaints the object's bounding box; up to 2 SD seeds per
   addition/substitution, then one retry pass of 3 seeds for invalid additions (seed offset 1000);
   recolor validity depends only on the object's saturation, so all three severities stay together.
4. **Metric change made on dev only**: SHAFE-DINO was at chance on recolor on dev, so a calibrated
   colour map was added; top-k pooling did not beat softmax pooling, so softmax was kept. The metric
   was frozen (commit `5c27959`) before the single test evaluation.
5. **Restoration**: all test images, plus 1 edited version each, with Swin2SR, Real-ESRGAN and the
   SD ×4 upscaler.
6. **Known limitation**: 42 of 62 substitutions could not get a blur control within 0.1 dB
   (their PSNR is lower than any blur reaches). They keep their noise control.

## Scaling to 300 images (7 Nov 2026)

The professor asked for more images. Rules for the scale-up:

7. **Nothing is thrown away.** The 120 original images keep their dev/test split, edits, controls,
   restorations and scores. The pipeline skips finished work. A copy of all 120-image results is
   kept in `results/snapshot_120/` for the before/after comparison.
8. **180 new COCO val2017 images**, same selection rules (512×512 centre crop, target object
   3–25% of the image, ≥90% of it inside the crop), no duplicates: we walk the same seeded image
   order as the first run and skip images already in the set.
9. **New split**: the 180 new images are split 90 dev / 90 test with their own seeded shuffle,
   giving 150 dev / 150 test. `images.csv` gets a `batch` column (`original` / `new`).
10. **Metric design unchanged.** `sqa/metrics.py`, the colour map and the tuning grid are not
    touched. Only step 06 is re-run on the larger dev split, which recomputes calibration statistics,
    the chosen layers/temperature and the thresholds. The new `freeze.json` is committed before
    step 07 runs on test.
11. **Edits only for the new batch.** Step 02 runs with `--batch new`, so the original 120 images
    get no extra edits (for example a retried addition box) and their data stays exactly as before.
    Per-image random seeds still use the row index in `images.csv`, which did not change for the
    original rows. Same addition retry as before: first pass up to 2 seeds, then one retry pass of
    up to 3 seeds (seed offset 1000) for invalid additions of the new batch.
12. **Controls are resumable.** Step 03 now keeps existing controls and only processes new edits,
    so the 737 original controls were not rewritten (checked: identical rows). 97 of the 1,207
    new-edit controls could not be PSNR-matched within 0.1 dB and were dropped, as before.
13. **Restoration on all 150 test images** (3 families × 288 outputs). Diffusion took about 75 s per
    image on the M2, about 3.6 h for the 175 new jobs.
14. **Re-freeze on the 150-image dev split** (8 Nov 2026). `sqa/metrics.py` and the metric/tuning
    config are unchanged since tag `results-120`. Calibration now uses 1,500 blurred/noised dev
    references. The tuning grid picked exactly the same settings as before for every variant
    (all four DINOv2 layers; τ = 0.05 for SHAFE-DINO, τ = 0.5 for SemRestore, k = 1% for the
    top-k variant). Dev mean AUROC: SemRestore 0.978 → 0.973, SHAFE-DINO 0.873 → 0.861. The new
    `freeze.json` was committed before step 07 touched the test split.
15. **Test evaluation and comparison** (8 Nov 2026). `07` and `10` were re-run on the 150-image test
    split; `scripts/11_compare_scaling.py` computes the before/after (120 vs 300) and new-images-only
    tables with bootstrap CIs. The 120-image numbers it reports were checked against the published
    ones (identical). No conclusion changed; additions now tie SHAFE-DINO; the severity weakness is
    clearer (see RESULTS.md).
16. **Human study page rebuilt** from the 150-image test split (130 pairs; edit-vs-blur now has the
    full 50). No ratings had been collected, so nothing was lost; the old pair list is in git history.
