"""Glue between saved maps, the freeze file and evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .evaluation import auroc
from .utils import read_json, results_dir

BASE_METRICS = ["psnr", "ssim", "ms_ssim", "lpips", "dists", "dino_global", "shafe_resnet"]


def load_maps(res_dir, item_ids, key="dino") -> dict:
    out = {}
    for i in item_ids:
        with np.load(Path(res_dir) / "maps" / f"{i}.npz") as z:
            out[i] = z[key]
    return out


def dino_scores(maps: dict, variant: str, params: dict, stored_layers, stats=None, colors=None) -> dict:
    colors = colors or {}
    return {i: M.dino_variant_score(m, variant, params, stored_layers, stats, colors.get(i))[0]
            for i, m in maps.items()}


def badness_table(items: pd.DataFrame, scores: pd.DataFrame, derived: dict[str, dict]) -> pd.DataFrame:
    """One row per item; one BADNESS column per metric (higher = more suspected change)."""
    df = items.merge(scores, on="item_id", how="inner")
    out = df[items.columns.tolist()].copy()
    for m in BASE_METRICS:
        if m in df and df[m].notna().any():
            out[m] = M.to_badness(m, df[m])
    for m, vals in derived.items():
        out[m] = M.to_badness(m, out.item_id.map(vals))
    return out


def metric_columns(bad: pd.DataFrame):
    return [c for c in BASE_METRICS + list(M.DINO_VARIANTS) if c in bad.columns]


def detection_split(bad: pd.DataFrame, metric: str, edit_type=None, control_kinds=("control_blur", "control_noise")):
    """Positives = valid edits; negatives = their PSNR-matched faithful controls."""
    e = bad[bad.kind == "edit"]
    c = bad[bad.kind.isin(control_kinds)]
    if edit_type is not None:
        e, c = e[e.edit_type == edit_type], c[c.edit_type == edit_type]
    return e, c


def mean_auroc_over_types(bad: pd.DataFrame, metric: str) -> float:
    vals = []
    for t in bad[bad.kind == "edit"].edit_type.unique():
        e, c = detection_split(bad, metric, t)
        if len(e) and len(c):
            vals.append(auroc(e[metric], c[metric]))
    return float(np.nanmean(vals)) if vals else np.nan


# ----------------------------------------------------------------------------- freeze
def metrics_fingerprint(cfg: dict, images_csv) -> str:
    """Hash of everything that defines the metric: its code, its config and the dev/test split."""
    h = hashlib.sha256()
    h.update(Path(M.__file__).read_bytes())
    h.update(json.dumps(cfg["metrics"], sort_keys=True).encode())
    h.update(json.dumps(cfg["tuning"], sort_keys=True).encode())
    split = pd.read_csv(images_csv)[["image_id", "split"]].sort_values("image_id")
    h.update(split.to_csv(index=False).encode())
    return h.hexdigest()


def load_freeze(cfg: dict, images_csv) -> dict:
    """Refuse to touch held-out test results unless the metric was frozen and is unchanged."""
    path = results_dir(cfg) / "freeze.json"
    if not path.exists():
        raise SystemExit("No results/freeze.json. Run scripts/06_freeze.py on the dev split first.")
    fz = read_json(path)
    if fz["fingerprint"] != metrics_fingerprint(cfg, images_csv):
        raise SystemExit("Metric code/config or the split changed after freezing. "
                         "Re-run 06_freeze.py (on dev) before evaluating on test.")
    return fz
