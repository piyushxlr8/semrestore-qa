"""Plot helpers shared by the scripts."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluation import upsample  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# readable, colour-blind-safe palette (Okabe-Ito)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

PRETTY = {"psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM", "lpips": "LPIPS", "dists": "DISTS",
          "dino_global": "DINO global", "shafe_resnet": "SHAFE-ResNet (official)",
          "shafe_dino": "SHAFE-DINO", "shafe_dino_calib": "SHAFE-DINO + calib",
          "dino_topk": "DINO + top-k", "semrestore": "SemRestore (ours)"}


def pretty(m):
    return PRETTY.get(m, m)


def mask_outline(ax, mask, color="#00ff66"):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.2)


def show(ax, img, title=None):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)


def show_heatmap(ax, img, hm, title=None, mask=None):
    hm = upsample(hm, img.shape[0])
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    show(ax, img * 0.45, title)
    ax.imshow(hm, cmap="inferno", alpha=0.65, vmin=0, vmax=1)
    if mask is not None:
        mask_outline(ax, mask)


def contact_sheet(rows, col_titles, path, cell=2.2):
    """rows: list of lists of images (None = empty cell)."""
    n_r, n_c = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_r, n_c, figsize=(cell * n_c, cell * n_r), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_c):
            ax = axes[r][c]
            img = row[c] if c < len(row) else None
            if img is None:
                ax.axis("off")
                continue
            show(ax, img, col_titles[c] if r == 0 else None)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
