"""Evaluation statistics. Everything works on BADNESS (higher = more suspected content change)."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


# ----------------------------------------------------------------------------- detection
def auroc(pos, neg) -> float:
    """P(badness of a random edit > badness of a random faithful control); ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = sps.rankdata(np.concatenate([pos, neg]))           # average ranks handle ties
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def best_f1_threshold(pos, neg) -> float:
    """Threshold (flag if badness > thr) that maximises F1. Fit on DEV only."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    cands = np.unique(np.concatenate([pos, neg]))
    best_t, best_f1 = cands[0], -1.0
    for t in np.concatenate([[cands[0] - 1e-9], cands]):
        f1 = f1_at(pos, neg, t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def f1_at(pos, neg, thr) -> float:
    tp = np.sum(np.asarray(pos) > thr)
    fp = np.sum(np.asarray(neg) > thr)
    fn = len(pos) - tp
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def fpr_threshold(neg, target_fpr=0.05) -> float:
    """Threshold so that only `target_fpr` of faithful controls are flagged. Fit on DEV only."""
    return float(np.quantile(np.asarray(neg, float), 1 - target_fpr))


def fnr_at(pos, thr) -> float:
    """Fraction of real edits the metric MISSES (badness not above the threshold)."""
    return float(np.mean(np.asarray(pos, float) <= thr))


def cohens_d(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                 / max(len(pos) + len(neg) - 2, 1))
    return float((pos.mean() - neg.mean()) / sp) if sp > 0 else np.nan


def win_rate(edit_bad, control_bad) -> float:
    """Paired: fraction of (edit, matched control) pairs where the metric correctly says the
    control is closer to the reference. 0.5 = coin flip. Ties count 1/2."""
    e, c = np.asarray(edit_bad, float), np.asarray(control_bad, float)
    return float(np.mean((c < e) + 0.5 * (c == e)))


def spearman(x, y) -> float:
    return float(sps.spearmanr(x, y).statistic) if len(x) > 2 else np.nan


def pearson(x, y) -> float:
    return float(sps.pearsonr(x, y).statistic) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_ci(df, stat_fn, group_col="image_id", iters=1000, seed=0, alpha=0.05):
    """95% CI by resampling whole IMAGES (items from the same image are not independent)."""
    groups = df.groupby(group_col).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        v = stat_fn(df.iloc[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ----------------------------------------------------------------------------- localization
def upsample(hm: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return cv2.resize(hm.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def pointing_game(hm, mask) -> float:
    """1 if the hottest pixel of the heatmap lies inside the true edit region."""
    hm = upsample(hm, mask.shape[0])
    return float(mask.ravel()[int(np.argmax(hm))])


def iou_at_mask_area(hm, mask) -> float:
    """Threshold-free IoU: mark the hottest N pixels, where N = true mask area."""
    hm = upsample(hm, mask.shape[0]).ravel()
    n = int(mask.sum())
    if n == 0:
        return np.nan
    pred = np.zeros_like(hm, bool)
    pred[np.argsort(hm)[::-1][:n]] = True
    m = mask.ravel()
    return float((pred & m).sum() / (pred | m).sum())


def pixel_auroc(hm, mask) -> float:
    hm = upsample(hm, mask.shape[0]).ravel()
    m = mask.ravel()
    return auroc(hm[m], hm[~m])


# ----------------------------------------------------------------------------- sharpness bias
def sharpness_bias_curve(edit_bad_by_image, blur_bad_by_image_sigma, sigmas):
    """Win-rate at each blur level: is the *blurred clean* reference judged better than the edit?
    Returns (list of win-rates, area under curve normalised to [0, 1])."""
    wins = []
    for s in sigmas:
        e, c = [], []
        for img, edits in edit_bad_by_image.items():
            b = blur_bad_by_image_sigma.get((img, s))
            if b is None:
                continue
            e.extend(edits)
            c.extend([b] * len(edits))
        wins.append(win_rate(e, c) if e else np.nan)
    sig = np.asarray(sigmas, float)
    w = np.asarray(wins, float)
    area = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(w, sig) / (sig[-1] - sig[0])) if len(sig) > 1 else float(w[0])
    return wins, area


# ----------------------------------------------------------------------------- human agreement
def krippendorff_alpha_nominal(ratings: np.ndarray) -> float:
    """ratings: (n_units, n_raters) array of category codes, np.nan for missing."""
    ratings = np.asarray(ratings, float)
    cats = np.unique(ratings[~np.isnan(ratings)])
    index = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))           # coincidence matrix
    for unit in ratings:
        vals = unit[~np.isnan(unit)]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[index[vals[i]], index[vals[j]]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if n <= 1:
        return np.nan
    d_o = (n - np.trace(o)) / n
    d_e = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else np.nan
