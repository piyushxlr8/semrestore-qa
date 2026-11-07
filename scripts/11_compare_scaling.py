"""11 - Before/after comparison for the scale-up (120 -> 300 images). Reads saved results only.

Computes, with 95% bootstrap CIs over images, on the held-out test split:
  * paired win-rate vs blur- and noise-matched controls, per metric
  * detection AUROC per edit type (edits vs matched controls)
  * FNR at the 5% FPR threshold that was frozen on dev
for three sets:
  before   : the 120-image run   (results/snapshot_120/badness.csv + its freeze.json)
  after    : the 300-image run   (results/badness.csv + results/freeze.json)
  new_only : the 300-image run restricted to the 90 brand-new test images

Output: results/tables/s1_before_after.csv, s2_new_test_only.csv, scaling_summary.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa import evaluation as V  # noqa: E402
from sqa.scoring import load_freeze  # noqa: E402
from sqa.utils import data_dir, ensure, load_config, read_json, results_dir  # noqa: E402
from sqa.viz import pretty  # noqa: E402

CONTROLS = ["control_blur", "control_noise"]
METRICS = ["psnr", "ssim", "lpips", "dists", "shafe_resnet", "shafe_dino", "semrestore"]
TYPES = ["addition", "removal", "substitution", "recolor"]


def headline(bad: pd.DataFrame, fz: dict, iters: int) -> pd.DataFrame:
    """bad = test-split badness rows (edits + controls). One row per (measure, metric, subset)."""
    out = []
    edits = bad[bad.kind == "edit"].set_index("edit_id")
    for m in [m for m in METRICS if m in bad.columns]:
        # paired win-rate
        for ctrl in CONTROLS:
            c = bad[bad.kind == ctrl].set_index("edit_id")
            common = edits.index.intersection(c.index)
            p = pd.DataFrame({"image_id": edits.loc[common, "image_id"].values,
                              "e": edits.loc[common, m].values, "c": c.loc[common, m].values})
            lo, hi = V.bootstrap_ci(p, lambda d: V.win_rate(d.e, d.c), iters=iters)
            out.append(dict(measure=f"win_rate_{ctrl.replace('control_', '')}", metric=m, subset="all",
                            value=V.win_rate(p.e, p.c), lo=lo, hi=hi, n=len(p)))
        # AUROC per edit type
        for t in TYPES:
            sub = bad[bad.kind.isin(["edit"] + CONTROLS) & (bad.edit_type == t)]
            pos, neg = sub[sub.kind == "edit"][m], sub[sub.kind.isin(CONTROLS)][m]
            if len(pos) < 2 or len(neg) < 2:
                continue
            lo, hi = V.bootstrap_ci(sub, lambda d: V.auroc(d[d.kind == "edit"][m], d[d.kind.isin(CONTROLS)][m]),
                                    iters=iters)
            out.append(dict(measure="auroc", metric=m, subset=t, value=V.auroc(pos, neg), lo=lo, hi=hi,
                            n=len(pos)))
        # FNR at the frozen 5%-FPR threshold
        thr = fz["thresholds"][m][f"fpr_{fz['fpr_target']}"]
        e = bad[bad.kind == "edit"]
        lo, hi = V.bootstrap_ci(e, lambda d: V.fnr_at(d[m], thr), iters=iters)
        out.append(dict(measure="fnr_at_5fpr", metric=m, subset="all", value=V.fnr_at(e[m], thr), lo=lo,
                        hi=hi, n=len(e)))
    return pd.DataFrame(out)


def fmt(r):
    return f"{r.value:.2f} [{r.lo:.2f}, {r.hi:.2f}]" if np.isfinite(r.value) else "n/a"


def main(cfg):
    res = results_dir(cfg)
    iters = cfg["evaluation"]["bootstrap_iters"]
    fz_after = load_freeze(cfg, data_dir(cfg) / "images.csv")         # refuses if not frozen
    snap = res / "snapshot_120"
    before = pd.read_csv(snap / "badness.csv")
    after = pd.read_csv(res / "badness.csv")
    images = pd.read_csv(data_dir(cfg) / "images.csv").set_index("image_id")
    after["batch"] = after.image_id.map(images.batch)

    sets = {"before (120 images, 60 test)": (before[before.split == "test"], read_json(snap / "freeze.json")),
            "after (300 images, 150 test)": (after[after.split == "test"], fz_after),
            "new test images only (90)": (after[(after.split == "test") & (after.batch == "new")], fz_after)}
    tables = {k: headline(b, f, iters) for k, (b, f) in sets.items()}
    for k, t in tables.items():
        t.insert(0, "set", k)
    allt = pd.concat(tables.values(), ignore_index=True)
    tdir = ensure(res / "tables")
    allt[allt.set != "new test images only (90)"].to_csv(tdir / "s1_before_after.csv", index=False)
    allt[allt.set == "new test images only (90)"].to_csv(tdir / "s2_new_test_only.csv", index=False)

    names = list(tables)
    piv = allt.assign(cell=allt.apply(fmt, axis=1)).pivot_table(
        index=["measure", "subset", "metric"], columns="set", values="cell", aggfunc="first")[names]
    lines = ["# Scale-up: 120 vs 300 images (held-out test split, 95% bootstrap CIs over images)", ""]
    sections = [("Win-rate vs blur-matched control", "win_rate_blur", "all", METRICS),
                ("Win-rate vs noise-matched control", "win_rate_noise", "all", METRICS),
                ("FNR at the dev-frozen 5% FPR threshold (lower is better)", "fnr_at_5fpr", "all", METRICS)]
    sections += [(f"AUROC: {t}", "auroc", t, ["shafe_dino", "semrestore"]) for t in TYPES]
    for title, meas, subset, ms in sections:
        lines += [f"## {title}", "", "| Metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for m in ms:
            if (meas, subset, m) in piv.index:
                row = piv.loc[(meas, subset, m)]
                lines.append(f"| {pretty(m)} | " + " | ".join(str(row[n]) for n in names) + " |")
        lines.append("")
    (tdir / "scaling_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
