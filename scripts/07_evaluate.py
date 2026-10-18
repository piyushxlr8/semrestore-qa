"""07 - Final evaluation on the HELD-OUT test split with the frozen metric.

Tables (results/tables/):
  t1_detection.csv         AUROC [95% CI], F1, FNR at 5% FPR, Cohen's d   (edits vs matched controls)
  t2_winrate.csv           paired win-rate: "blurry/noisy but correct" judged better than the edit?
  t3_sharpness_bias.csv    win-rate as the clean reference is blurred more and more (SBC)
  t4_severity.csv          Spearman / Pearson between badness and edit severity
  t5_localization.csv      does the heatmap point at the edited region?
  t6_cross_restoration.csv edit-then-restore AUROC per restoration family (+ real-output stats)
  t7_ablation.csv          what each component of our metric adds
  summary.md               one headline table for the slides
Synthetic edits and real restorer outputs are always reported in separate tables.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa import metrics as M  # noqa: E402
from sqa import evaluation as V  # noqa: E402
from sqa.scoring import badness_table, dino_scores, load_freeze, load_maps, metric_columns  # noqa: E402
from sqa.utils import data_dir, ensure, load_config, load_mask, results_dir  # noqa: E402
from sqa.viz import pretty  # noqa: E402

CONTROLS = ["control_blur", "control_noise"]


def fmt_ci(v, lo, hi):
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(v) else "n/a"


def main(cfg):
    res = results_dir(cfg)
    fz = load_freeze(cfg, data_dir(cfg) / "images.csv")
    iters = cfg["evaluation"]["bootstrap_iters"]
    items = pd.read_csv(res / "items.csv")
    scores = pd.read_csv(res / "scores.csv")
    stored, stats, params = fz["stored_layers"], fz["calibration"], fz["params"]

    # ---- badness for every item with the FROZEN settings (dev + test; only test is reported)
    maps = load_maps(res, items.item_id)
    derived = {v: dino_scores(maps, v, params[v], stored, stats) for v in params}
    bad_all = badness_table(items, scores, derived)
    bad_all.to_csv(res / "badness.csv", index=False)
    bad = bad_all[bad_all.split == "test"]
    metrics = metric_columns(bad)
    tdir = ensure(res / "tables")
    edit_types = ["all"] + sorted(bad[bad.kind == "edit"].edit_type.unique())
    print(f"test items: {bad.kind.value_counts().to_dict()}")

    # ---- T1 detection --------------------------------------------------------------------
    t1 = []
    for m in metrics:
        thr = fz["thresholds"][m]
        for t in edit_types:
            sub = bad[bad.kind.isin(["edit"] + CONTROLS)]
            if t != "all":
                sub = sub[sub.edit_type == t]
            pos, neg = sub[sub.kind == "edit"][m], sub[sub.kind.isin(CONTROLS)][m]
            if len(pos) < 2 or len(neg) < 2:
                continue
            a = V.auroc(pos, neg)
            lo, hi = V.bootstrap_ci(sub, lambda d: V.auroc(d[d.kind == "edit"][m], d[d.kind.isin(CONTROLS)][m]),
                                    iters=iters)
            t1.append(dict(data_source="synthetic_edit", metric=m, edit_type=t, n_pos=len(pos), n_neg=len(neg),
                           n_images=sub.image_id.nunique(), auroc=a, auroc_lo=lo, auroc_hi=hi,
                           f1=V.f1_at(pos, neg, thr["f1"]),
                           fnr_at_5fpr=V.fnr_at(pos, thr[f"fpr_{fz['fpr_target']}"]),
                           fpr_at_5fpr_thr=float(np.mean(neg > thr[f"fpr_{fz['fpr_target']}"])),
                           cohens_d=V.cohens_d(pos, neg)))
    t1 = pd.DataFrame(t1)
    t1.to_csv(tdir / "t1_detection.csv", index=False)

    # ---- T2 paired win-rate vs PSNR-matched controls -------------------------------------
    edits = bad[bad.kind == "edit"].set_index("edit_id")
    t2 = []
    for ctrl in CONTROLS:
        c = bad[bad.kind == ctrl].set_index("edit_id")
        common = edits.index.intersection(c.index)
        paired = pd.DataFrame({"image_id": edits.loc[common, "image_id"].values,
                               "edit_type": edits.loc[common, "edit_type"].values})
        for m in metrics:
            paired["e"], paired["c"] = edits.loc[common, m].values, c.loc[common, m].values
            for t in edit_types:
                p = paired if t == "all" else paired[paired.edit_type == t]
                if len(p) < 2:
                    continue
                w = V.win_rate(p.e, p.c)
                lo, hi = V.bootstrap_ci(p.reset_index(drop=True), lambda d: V.win_rate(d.e, d.c), iters=iters)
                t2.append(dict(data_source="synthetic_edit", metric=m, control=ctrl.replace("control_", ""),
                               edit_type=t, n_pairs=len(p), win_rate=w, lo=lo, hi=hi))
    t2 = pd.DataFrame(t2)
    t2.to_csv(tdir / "t2_winrate.csv", index=False)

    # ---- T3 sharpness bias curve ---------------------------------------------------------
    sigmas = cfg["evaluation"]["sbc_sigmas"]
    deg = bad[(bad.kind == "degraded") & (bad.edit_type == "blur")]
    t3 = []
    for m in metrics:
        eb = bad[bad.kind == "edit"].groupby("image_id")[m].apply(list).to_dict()
        bb = {(r.image_id, float(r.param)): getattr(r, m) for r in deg.itertuples()}
        wins, area = V.sharpness_bias_curve(eb, bb, [float(s) for s in sigmas])
        t3.append(dict(metric=m, sbc_area=area, **{f"win@sigma{s}": w for s, w in zip(sigmas, wins)}))
    t3 = pd.DataFrame(t3)
    t3.to_csv(tdir / "t3_sharpness_bias.csv", index=False)

    # ---- T4 severity ---------------------------------------------------------------------
    t4 = []
    e_all = bad[bad.kind == "edit"]
    for m in metrics:
        for t in edit_types[1:]:
            g = e_all[e_all.edit_type == t]
            if len(g) < 5 or g.severity.nunique() < 2:
                continue
            lo, hi = V.bootstrap_ci(g, lambda d: V.spearman(d[m], d.severity), iters=iters)
            t4.append(dict(metric=m, edit_type=t, severity="hue shift" if t == "recolor" else "area fraction",
                           n=len(g), spearman=V.spearman(g[m], g.severity), lo=lo, hi=hi,
                           pearson=V.pearson(g[m], g.severity)))
    t4 = pd.DataFrame(t4)
    t4.to_csv(tdir / "t4_severity.csv", index=False)

    # ---- T5 localization (metrics with a heatmap) ----------------------------------------
    loc_rows = []
    smaps = load_maps(res, e_all.item_id, key="shafe_resnet")
    for r in e_all.itertuples():
        mask = load_mask(r.mask_path)
        heat = {"shafe_resnet": smaps[r.item_id]}
        for v in ["shafe_dino", "semrestore"]:
            cal = M.DINO_VARIANTS[v][0]
            heat[v] = M.dino_heatmap(maps[r.item_id], stored, params[v]["layers"], cal, stats)
        for m, hm in heat.items():
            loc_rows.append(dict(metric=m, edit_type=r.edit_type, image_id=r.image_id,
                                 pointing=V.pointing_game(hm, mask), iou=V.iou_at_mask_area(hm, mask),
                                 pixel_auroc=V.pixel_auroc(hm, mask)))
    loc = pd.DataFrame(loc_rows)
    t5 = pd.concat([loc.groupby("metric")[["pointing", "iou", "pixel_auroc"]].mean().assign(edit_type="all"),
                    loc.groupby(["metric", "edit_type"])[["pointing", "iou", "pixel_auroc"]].mean()
                    .reset_index("edit_type")]).reset_index()
    t5.to_csv(tdir / "t5_localization.csv", index=False)

    # ---- T6 cross-restoration (real restorers; reported separately) ----------------------
    t6 = []
    rest = bad[bad.kind.isin(["restored_ref", "restored_edit"])]
    for fam, g in rest.groupby("family"):
        for m in metrics:
            pos, neg = g[g.kind == "restored_edit"][m], g[g.kind == "restored_ref"][m]
            if len(pos) < 2 or len(neg) < 2:
                continue
            lo, hi = V.bootstrap_ci(g, lambda d: V.auroc(d[d.kind == "restored_edit"][m],
                                                         d[d.kind == "restored_ref"][m]), iters=iters)
            t6.append(dict(data_source="real_restorer", family=fam, metric=m, n_pos=len(pos), n_neg=len(neg),
                           auroc=V.auroc(pos, neg), lo=lo, hi=hi,
                           mean_badness_clean_restoration=float(neg.mean())))
    t6 = pd.DataFrame(t6)
    t6.to_csv(tdir / "t6_cross_restoration.csv", index=False)

    # ---- T7 ablation ---------------------------------------------------------------------
    t7 = []
    for v in M.DINO_VARIANTS:
        row = dict(variant=v, calibration=M.DINO_VARIANTS[v][0], pooling=M.DINO_VARIANTS[v][1],
                   frozen_params=str(params[v]), dev_mean_auroc=fz["dev_mean_auroc"][v])
        r1 = t1[(t1.metric == v) & (t1.edit_type == "all")]
        row["test_auroc"] = float(r1.auroc.iloc[0]) if len(r1) else np.nan
        for ctrl in ["blur", "noise"]:
            r2 = t2[(t2.metric == v) & (t2.control == ctrl) & (t2.edit_type == "all")]
            row[f"winrate_{ctrl}"] = float(r2.win_rate.iloc[0]) if len(r2) else np.nan
        r6 = t6[t6.metric == v] if len(t6) else t6
        row["cross_restoration_auroc"] = float(r6.auroc.mean()) if len(r6) else np.nan
        t7.append(row)
    t7 = pd.DataFrame(t7)
    t7.to_csv(tdir / "t7_ablation.csv", index=False)

    # ---- summary -------------------------------------------------------------------------
    lines = ["# SemRestore-QA: headline results (held-out test split)", "",
             "Synthetic edits vs PSNR-matched faithful controls. Higher is better except FNR.", "",
             "| Metric | AUROC [95% CI] | Win-rate vs blur | Win-rate vs noise | FNR @5% FPR | SBC | "
             "Severity ρ (area) | Pointing | Cross-restoration AUROC* |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        r1 = t1[(t1.metric == m) & (t1.edit_type == "all")]
        wb = t2[(t2.metric == m) & (t2.control == "blur") & (t2.edit_type == "all")]
        wn = t2[(t2.metric == m) & (t2.control == "noise") & (t2.edit_type == "all")]
        sb = t3[t3.metric == m]
        sv = t4[(t4.metric == m) & (t4.edit_type != "recolor")] if len(t4) else t4
        pt = t5[(t5.metric == m) & (t5.edit_type == "all")] if len(t5) else t5
        cr = t6[t6.metric == m] if len(t6) else t6
        cell = lambda df, col: f"{df[col].iloc[0]:.3f}" if len(df) else "–"  # noqa: E731
        lines.append(f"| {pretty(m)} | {fmt_ci(r1.auroc.iloc[0], r1.auroc_lo.iloc[0], r1.auroc_hi.iloc[0]) if len(r1) else '–'} "
                     f"| {cell(wb, 'win_rate')} | {cell(wn, 'win_rate')} | {cell(r1, 'fnr_at_5fpr')} "
                     f"| {cell(sb, 'sbc_area')} | {f'{sv.spearman.mean():.3f}' if len(sv) else '–'} "
                     f"| {cell(pt, 'pointing')} | {f'{cr.auroc.mean():.3f}' if len(cr) else '–'} |")
    lines += ["", "*Real restorer outputs (edit-then-restore), mean over families; see t6 for each family.",
              "Win-rate 0.5 = coin flip. Matched pairs differ by at most 0.1 dB PSNR, so PSNR's win-rate there is essentially arbitrary."]
    (tdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
