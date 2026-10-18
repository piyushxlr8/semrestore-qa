"""Fast CPU tests (no downloads). Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa import edits as E  # noqa: E402
from sqa import evaluation as V  # noqa: E402
from sqa import metrics as M  # noqa: E402
from sqa.controls import match_blur, match_noise  # noqa: E402
from sqa.utils import dilate, paste_back, psnr, quantize  # noqa: E402


def textured(seed=0, size=128):
    rng = np.random.default_rng(seed)
    import cv2
    return np.clip(cv2.GaussianBlur(rng.random((size, size, 3)).astype(np.float32), (0, 0), 2) * 1.5 - 0.2, 0, 1)


# ----------------------------------------------------------------------------- data rules
def test_paste_back_keeps_outside_pixels_exact():
    ref, gen = textured(0), textured(1)
    region = np.zeros(ref.shape[:2], bool)
    region[30:70, 40:90] = True
    out = paste_back(ref, gen, region, feather_px=3)
    assert np.array_equal(out[~region], ref[~region])          # mask is a valid ground truth
    assert np.abs(out[region] - ref[region]).mean() > 0.05


def test_recolor_only_changes_masked_pixels():
    ref = textured(2)
    mask = np.zeros(ref.shape[:2], bool)
    mask[10:50, 10:50] = True
    out = E.recolor(ref, mask, 120)
    assert np.allclose(out[~mask], ref[~mask], atol=1e-5)
    assert np.abs(out[mask] - ref[mask]).mean() > 0.01


def test_addition_box_avoids_objects():
    rng = np.random.default_rng(0)
    objs = np.zeros((256, 256), bool)
    objs[:, :128] = True
    box = E.find_addition_box(objs, 0.05, rng)
    assert box is not None and (box & objs).mean() / box.mean() < 0.05 + 1e-9


def test_box_around_covers_mask():
    mask = np.zeros((64, 64), bool)
    mask[10:20, 30:35] = True
    mask[25, 40] = True
    box = E.box_around(mask)
    assert box[mask].all() and box.sum() == (26 - 10) * (41 - 30)


def test_odds_drop():
    assert E.odds_drop(0.999, 0.94) > 5          # removed horse next to its cart: valid
    assert E.odds_drop(0.99, 0.98) < 5           # LaMa failed, object still there: invalid
    assert abs(E.odds_drop(0.5, 0.5) - 1) < 1e-9


@pytest.mark.parametrize("target", [25.0, 30.0, 35.0])
def test_psnr_matched_controls(target):
    ref = quantize(textured(3))
    _, img, gap = match_blur(ref, target, tol=0.1)
    assert gap <= 0.1 and abs(psnr(img, ref) - target) <= 0.1
    _, img, gap = match_noise(ref, target, tol=0.1)
    assert gap <= 0.1


def test_dilate_grows_mask():
    m = np.zeros((50, 50), bool)
    m[25, 25] = True
    assert dilate(m, 3).sum() > 1 and dilate(m, 3)[25, 28]


# ----------------------------------------------------------------------------- metric rules
def test_badness_direction():
    assert M.to_badness("psnr", 40) < M.to_badness("psnr", 20)      # higher PSNR = less bad
    assert M.to_badness("lpips", 0.1) < M.to_badness("lpips", 0.5)


def test_encoder_families_are_separate():
    """Circularity guard: nothing that MAKES or CHECKS edits shares a family with a metric."""
    assert not (set(M.METRIC_ENCODER_FAMILY.values()) & M.GENERATOR_FAMILIES)
    assert {E.SDInpainter.family, E.LamaInpainter.family, E.ClipChecker.family} <= M.GENERATOR_FAMILIES


def test_pooling():
    m = np.zeros((10, 10))
    m[0, 0] = 1.0
    assert M.topk_pool(m, 0.01) == 1.0                 # one bad patch is not averaged away
    assert M.softmax_pool(m, 0.01) > 0.99
    assert M.softmax_pool(m, 100.0) < 0.05             # large tau ~ mean pooling


def test_calibration_and_variants():
    rng = np.random.default_rng(0)
    calib = [rng.random((4, 8, 8)) * 0.1 for _ in range(5)]
    calib_col = [rng.random((8, 8)) for _ in range(5)]
    stats = M.calibration_stats(calib, calib_col)
    edit = rng.random((4, 8, 8)) * 0.1
    edit[:, 2:4, 2:4] = 0.9
    for v in M.DINO_VARIANTS:
        params = {"layers": [6, 12], "tau": 0.05, "k_frac": 0.05}
        s_edit, hm = M.dino_variant_score(edit, v, params, [3, 6, 9, 12], stats, calib_col[1])
        s_ctrl, _ = M.dino_variant_score(calib[0], v, params, [3, 6, 9, 12], stats, calib_col[0])
        assert hm.shape == (8, 8) and s_edit > s_ctrl


def test_colour_map_sees_recolour_but_not_blur():
    from sqa.utils import gaussian_blur
    ref = textured(5, 224)
    mask = np.zeros(ref.shape[:2], bool)
    mask[40:120, 40:120] = True
    recol = M.color_map(ref, E.recolor(ref, mask, 150), 16)
    blur = M.color_map(ref, gaussian_blur(ref, 2.0), 16)
    assert recol.max() > 5 * blur.max()
    assert recol[3:7, 3:7].mean() > 10 * recol[10:, 10:].mean()      # only the recoloured patches


def test_identity_gives_best_score_with_random_networks():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    cfg = {"metrics": {"fake_models": True, "batch_size": 2,
                       "shafe_resnet": {"model": "resnetaa50d.d_in12k", "out_indices": [1, 2],
                                        "lowpass_radius_at_256": 60, "temperature": 0.02},
                       "dino": {"model": "x", "input_size": 112, "layers": [1, 2], "lowpass_radius_at_256": 60}}}
    scorer = M.RawScorer(cfg, "cpu")
    ref, other = textured(4, 192), textured(5, 192)
    rows, maps = scorer.score_batch([ref, ref], [ref, other])
    same, diff = rows
    for m in ["ssim", "ms_ssim", "shafe_resnet", "dino_global"]:   # (random LPIPS weights can be negative)
        assert M.to_badness(m, same[m]) <= M.to_badness(m, diff[m]) + 1e-6, m
    assert maps[0]["dino"].shape == (2, 8, 8)  # 112 / 14 = 8
    assert np.abs(maps[0]["dino"]).max() < 1e-2      # identical images -> zero distance


# ----------------------------------------------------------------------------- statistics
def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 5, 40).astype(float), rng.integers(0, 4, 60).astype(float)
    y = np.r_[np.ones(40), np.zeros(60)]
    assert V.auroc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))


def test_thresholds_and_fnr():
    neg = np.arange(100.0)
    thr = V.fpr_threshold(neg, 0.05)
    assert np.mean(neg > thr) == pytest.approx(0.05, abs=0.011)
    assert V.fnr_at([0, 1, 200, 300], thr) == 0.5
    assert V.f1_at([5, 6], [1, 2], V.best_f1_threshold([5, 6], [1, 2])) == 1.0


def test_win_rate_and_sbc():
    assert V.win_rate([2, 2, 2], [1, 3, 2]) == pytest.approx(0.5)
    wins, area = V.sharpness_bias_curve({"a": [1.0]}, {("a", 1.0): 0.5, ("a", 2.0): 2.0}, [1.0, 2.0])
    assert wins == [1.0, 0.0] and area == pytest.approx(0.5)


def test_bootstrap_resamples_images_not_rows():
    df = pd.DataFrame({"image_id": ["a"] * 50 + ["b"], "x": [1.0] * 50 + [0.0]})
    lo, hi = V.bootstrap_ci(df, lambda d: d.x.mean(), iters=300)
    assert lo == 0.0 and hi == 1.0                   # only 2 independent images -> huge CI


def test_krippendorff_alpha_matches_reference_values():
    perfect = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.0]])
    assert V.krippendorff_alpha_nominal(perfect) == pytest.approx(1.0)
    # value computed with the `krippendorff` package (nominal) for this matrix
    m = np.array([[0, 0, 1], [1, 1, 1], [2, 2, 0], [0, np.nan, 0], [1, 2, 1]], float)
    assert V.krippendorff_alpha_nominal(m) == pytest.approx(0.3809523810, abs=1e-6)


def test_localization_scores():
    mask = np.zeros((64, 64), bool)
    mask[:32, :32] = True
    hm = np.zeros((8, 8))
    hm[:4, :4] = 1
    assert V.pointing_game(hm, mask) == 1.0
    assert V.iou_at_mask_area(hm, mask) > 0.8
    assert V.pixel_auroc(hm, mask) > 0.9


# ----------------------------------------------------------------------------- human study
def test_vote_table_and_agreement():
    from sqa.human import split_half_agreement, vote_table
    r = pd.DataFrame({"rater": list("xxyyzz"), "pair_id": ["p1", "p2"] * 3,
                      "choice": ["a", "b", "a", "b", "a", "same"]})
    v = vote_table(r).set_index("pair_id")
    assert v.loc["p1", "majority"] == "a" and v.loc["p1", "share_a"] == 1.0
    assert v.loc["p2", "majority"] == "b"
    assert split_half_agreement(r, n_splits=20) == 1.0
