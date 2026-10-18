"""10 - Figures for the report and slides (all from saved results; no models needed).

  fig1_examples.png            reference / edit / same-PSNR blur control / heatmaps
  fig2_winrate.png             the key test: is "blurry but correct" judged better than "sharp but wrong"?
  fig3_sharpness_bias.png      win-rate as the clean image gets blurrier (SBC)
  fig4_cross_restoration.png   edit-then-restore AUROC per restoration family
  fig5_restorers.png           what the three restoration families actually output
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa import metrics as M  # noqa: E402
from sqa.scoring import load_freeze  # noqa: E402
from sqa.utils import data_dir, ensure, load_config, load_image, load_mask, results_dir  # noqa: E402
from sqa.viz import PALETTE, mask_outline, plt, pretty, show, show_heatmap  # noqa: E402

KEY = ["psnr", "ssim", "lpips", "dists", "shafe_resnet", "shafe_dino", "semrestore"]


def main(cfg):
    res = results_dir(cfg)
    fz = load_freeze(cfg, data_dir(cfg) / "images.csv")
    fdir = ensure(res / "figures")
    bad = pd.read_csv(res / "badness.csv")
    test = bad[bad.split == "test"]
    t2 = pd.read_csv(res / "tables" / "t2_winrate.csv")
    t3 = pd.read_csv(res / "tables" / "t3_sharpness_bias.csv")
    t6_path = res / "tables" / "t6_cross_restoration.csv"
    t6 = pd.read_csv(t6_path) if t6_path.exists() and t6_path.stat().st_size > 1 else pd.DataFrame()
    key = [m for m in KEY if m in test.columns]

    # ---- fig1: examples with heatmaps
    edits = test[test.kind == "edit"]
    ctrl = test[test.kind == "control_blur"].set_index("edit_id")
    picks = []
    for t in ["addition", "removal", "substitution", "recolor"]:
        g = edits[(edits.edit_type == t) & edits.edit_id.isin(ctrl.index)]
        if len(g):
            picks.append(g.sort_values("semrestore", ascending=False).iloc[len(g) // 3])
    if picks:
        cols = ["reference", "edit (content changed)", "blur control (same PSNR)", "SHAFE-ResNet map",
                "SemRestore map"]
        fig, axes = plt.subplots(len(picks), 5, figsize=(12, 2.5 * len(picks)), squeeze=False)
        for r, e in enumerate(picks):
            ref, ed = load_image(e.ref_path), load_image(e.test_path)
            c = ctrl.loc[e.edit_id]
            mask = load_mask(e.mask_path)
            with np.load(res / "maps" / f"{e.item_id}.npz") as z:
                smap, dmap, cmap = z["shafe_resnet"], z["dino"], z["color"]
            ours = M.dino_heatmap(dmap, fz["stored_layers"], fz["params"]["semrestore"]["layers"], True,
                                  fz["calibration"], cmap)
            show(axes[r][0], ref, cols[0] if r == 0 else None)
            show(axes[r][1], ed, cols[1] if r == 0 else None)
            mask_outline(axes[r][1], mask)
            show(axes[r][2], load_image(c.test_path), cols[2] if r == 0 else None)
            show_heatmap(axes[r][3], ed, smap, cols[3] if r == 0 else None, mask)
            show_heatmap(axes[r][4], ed, ours, cols[4] if r == 0 else None, mask)
            axes[r][0].set_ylabel(e.edit_type, fontsize=10)
            verdict = lambda m: "✓" if c[m] < e[m] else "✗"  # noqa: E731
            axes[r][2].set_xlabel("  ".join(f"{pretty(m).split(' ')[0]} {verdict(m)}"
                                            for m in ["ssim", "lpips", "semrestore"] if m in e), fontsize=8)
        fig.suptitle("✓ = metric correctly says the blurry control is closer to the reference", fontsize=9, y=1.0)
        fig.tight_layout()
        fig.savefig(fdir / "fig1_examples.png", bbox_inches="tight")
        plt.close(fig)

    # ---- fig2: win-rate bars
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    for j, ctrl_type in enumerate(["blur", "noise"]):
        sub = t2[(t2.control == ctrl_type) & (t2.edit_type == "all")].set_index("metric").reindex(key)
        x = np.arange(len(key)) + (j - 0.5) * w
        ax.bar(x, sub.win_rate, w, yerr=[sub.win_rate - sub.lo, sub.hi - sub.win_rate], capsize=2,
               color=PALETTE[j], label=f"vs {ctrl_type}-matched control")
    ax.axhline(0.5, color="gray", lw=1, ls="--")
    ax.text(len(key) - 0.5, 0.51, "coin flip", ha="right", fontsize=8, color="gray")
    ax.set_xticks(np.arange(len(key)), [pretty(m) for m in key], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("win-rate (higher = better)")
    ax.set_title("Sharp-but-wrong vs blurry/noisy-but-correct at the SAME PSNR")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fdir / "fig2_winrate.png", bbox_inches="tight")
    plt.close(fig)

    # ---- fig3: sharpness-bias curves
    sig_cols = [c for c in t3.columns if c.startswith("win@sigma")]
    sigmas = [float(c.replace("win@sigma", "")) for c in sig_cols]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, m in enumerate(key):
        row = t3[t3.metric == m]
        if len(row):
            ax.plot(sigmas, row[sig_cols].iloc[0].values, marker="o", ms=3, color=PALETTE[k % len(PALETTE)],
                    label=f"{pretty(m)} (SBC {row.sbc_area.iloc[0]:.2f})", lw=2 if m == "semrestore" else 1.2)
    ax.set_xlabel("blur σ applied to the clean reference")
    ax.set_ylabel("P(blurred clean judged better than edit)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("Sharpness bias: higher curve = less fooled by sharpness")
    fig.tight_layout()
    fig.savefig(fdir / "fig3_sharpness_bias.png", bbox_inches="tight")
    plt.close(fig)

    # ---- fig4: cross restoration
    if len(t6):
        fams = [f for f in ["regression", "gan", "diffusion"] if f in set(t6.family)]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        w = 0.8 / len(fams)
        for j, fam in enumerate(fams):
            sub = t6[t6.family == fam].set_index("metric").reindex(key)
            x = np.arange(len(key)) + (j - (len(fams) - 1) / 2) * w
            ax.bar(x, sub.auroc, w, yerr=[sub.auroc - sub.lo, sub.hi - sub.auroc], capsize=2,
                   color=PALETTE[j + 2], label=fam)
        ax.axhline(0.5, color="gray", lw=1, ls="--")
        ax.set_xticks(np.arange(len(key)), [pretty(m) for m in key], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUROC")
        ax.set_title("Edit-then-restore: detecting content change in real restorer outputs")
        ax.legend(frameon=False, fontsize=8, title="restorer family", title_fontsize=8)
        fig.tight_layout()
        fig.savefig(fdir / "fig4_cross_restoration.png", bbox_inches="tight")
        plt.close(fig)

    # ---- fig5: restorer outputs for one image
    rest = test[test.kind == "restored_ref"]
    if len(rest):
        img = rest.image_id.value_counts().index[0]
        g = rest[rest.image_id == img].set_index("family")
        fams = [f for f in ["regression", "gan", "diffusion"] if f in g.index]
        restored_csv = pd.read_csv(data_dir(cfg) / "restored" / "restored.csv").set_index("restore_id")
        lr = load_image(restored_csv.loc[g.iloc[0].item_id, "lr_path"])
        fig, axes = plt.subplots(1, 2 + len(fams), figsize=(2.6 * (2 + len(fams)), 2.9))
        show(axes[0], load_image(g.iloc[0].ref_path), "reference (512)")
        show(axes[1], lr, "input (128, bicubic ↓4)")
        for k, fam in enumerate(fams):
            show(axes[2 + k], load_image(g.loc[fam].test_path), fam)
        fig.tight_layout()
        fig.savefig(fdir / "fig5_restorers.png", bbox_inches="tight")
        plt.close(fig)
    print(f"figures -> {fdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
