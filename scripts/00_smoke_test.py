"""00 - Smoke test: run the WHOLE pipeline on tiny synthetic images in a few minutes on CPU.

No downloads, no GPU. Uses random-weight networks and fake restorers, so the NUMBERS ARE
MEANINGLESS; it only proves the plumbing works (files, tables, figures, freeze guard, human study).

    python scripts/00_smoke_test.py            # writes everything to ./smoke_run/
"""
import argparse
import copy
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sqa import edits as E  # noqa: E402
from sqa.utils import dilate, ensure, paste_back, psnr, quantize, save_image, save_mask  # noqa: E402

COLORS = [(0.85, 0.2, 0.2), (0.2, 0.6, 0.9), (0.95, 0.75, 0.1), (0.3, 0.8, 0.3), (0.7, 0.3, 0.8)]


def run_script(name, cfg):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"\n================ {name} ================")
    mod.main(cfg)


def draw_shape(img, mask, kind, color, center, r):
    m = np.zeros(mask.shape, np.uint8)
    if kind == "circle":
        cv2.circle(m, center, r, 1, -1)
    else:
        cv2.rectangle(m, (center[0] - r, center[1] - r), (center[0] + r, center[1] + r), 1, -1)
    m = m.astype(bool)
    img[m] = color
    return m


def synthetic_dataset(root: Path, n=12, size=512, seed=0):
    rng = np.random.default_rng(seed)
    rows, edit_rows = [], []
    for i in range(n):
        yy, xx = np.mgrid[0:size, 0:size] / size
        base = np.stack([0.3 + 0.4 * yy, 0.4 + 0.2 * xx, 0.5 - 0.2 * yy], -1)
        tex = cv2.GaussianBlur(rng.random((size, size, 1)).astype(np.float32), (0, 0), 3)[..., None] * 0.3
        img = np.clip(base + tex - 0.15, 0, 1).astype(np.float32)
        allm = np.zeros((size, size), bool)
        obj = None
        for k in range(3):
            c = (int(rng.integers(100, 412)), int(rng.integers(100, 412)))
            m = draw_shape(img, allm, ["circle", "square"][k % 2], COLORS[k], c, int(rng.integers(45, 70)))
            allm |= m
            obj = m if k == 0 else obj
        image_id = f"synth{i:03d}"
        ref_path, op, ap = root / "refs" / f"{image_id}.png", root / "masks" / f"{image_id}_o.png", \
            root / "masks" / f"{image_id}_a.png"
        img = quantize(img)
        save_image(img, ref_path), save_mask(obj, op), save_mask(allm, ap)
        split = "dev" if i < n // 2 else "test"
        rows.append(dict(image_id=image_id, coco_id=i, ref_path=str(ref_path), object_category="circle",
                         object_mask_path=str(op), object_area_frac=obj.mean(), all_objects_mask_path=str(ap),
                         categories_present="circle;square", split=split))

        def add(etype, out, region, sev):
            eid = f"{image_id}_{etype}"
            ep, mp = root / "edits" / "images" / f"{eid}.png", root / "edits" / "masks" / f"{eid}.png"
            out = quantize(out)
            save_image(out, ep), save_mask(region, mp)
            edit_rows.append(dict(edit_id=eid, image_id=image_id, split=split, edit_type=etype.rstrip("0123456789"),
                                  severity=sev, mask_path=str(mp), edited_path=str(ep), psnr=psnr(out, img),
                                  mean_abs_change=float(np.abs(out - img)[region].mean()), valid=1,
                                  source_category="circle", target_category="", prompt="synthetic", seed=0,
                                  generator="synthetic", clip_prob=np.nan))

        region = dilate(obj, 10)
        removed = cv2.inpaint((img * 255).astype(np.uint8), region.astype(np.uint8), 5, cv2.INPAINT_TELEA) / 255.0
        add("removal", paste_back(img, removed.astype(np.float32), region, 3), region, region.mean())
        sub = img.copy()
        sub[region] = img[~allm].mean(0)
        draw_shape(sub, np.zeros_like(obj), "square", COLORS[4], tuple(int(v) for v in np.argwhere(obj).mean(0)[::-1]), 40)
        add("substitution", paste_back(img, sub, region, 3), region, region.mean())
        box = E.find_addition_box(dilate(allm, 8), 0.05, rng)
        if box is not None:
            addi = img.copy()
            ys, xs = np.where(box)
            draw_shape(addi, box, "circle", COLORS[3], (int(xs.mean()), int(ys.mean())), 30)
            add("addition", paste_back(img, addi, box, 3), box, box.mean())
        for shift in (30, 90, 150):
            add(f"recolor{shift}", E.recolor(img, obj, shift), obj, shift)
    pd.DataFrame(rows).to_csv(root / "images.csv", index=False)
    ensure(root / "edits")
    pd.DataFrame(edit_rows).to_csv(root / "edits" / "edits.csv", index=False)


def simulate_ratings(cfg, n_raters=4, seed=0):
    """SIMULATED raters (clearly labelled) just to exercise the analysis code."""
    res = Path(cfg["paths"]["results_dir"])
    pairs = pd.read_csv(res / "human" / "pairs.csv")
    rng = np.random.default_rng(seed)
    out = ensure(res / "human" / "ratings")
    for r in range(n_raters):
        rows = []
        for k, p in enumerate(pairs.itertuples()):
            good = {"gold": "a", "severity": "a", "edit_vs_blur": "b"}.get(p.pair_type, rng.choice(["a", "b"]))
            if r == n_raters - 1:          # one careless rater who should be excluded
                choice = rng.choice(["a", "b", "same"])
            else:
                choice = good if rng.random() < 0.8 else rng.choice(["a", "b", "same"])
            rows.append(dict(rater=f"SIMULATED_{r}", pair_id=p.pair_id, choice=choice, left_item="a",
                             rt_ms=int(rng.integers(800, 4000)), order=k))
        pd.DataFrame(rows).to_csv(out / f"ratings_SIMULATED_{r}.csv", index=False)


def main(out_dir="smoke_run"):
    out = Path(out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {"data_dir": str(out / "data"), "results_dir": str(out / "results")}
    cfg["metrics"]["fake_models"] = True
    cfg["restore"]["n_images"] = 6
    cfg["evaluation"]["bootstrap_iters"] = 50
    cfg["human"].update(n_edit_vs_blur=5, n_real_vs_real=5, n_severity=4, n_gold=3, display_size=256)
    cfg_path = out / "smoke_config.yaml"
    ensure(out)
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    cfg["_config_path"] = str(cfg_path)

    print("creating synthetic images and edits ...")
    synthetic_dataset(Path(cfg["paths"]["data_dir"]))
    for name in ["03_make_controls", "04_restore", "05_score", "06_freeze", "07_evaluate", "08_human_pairs"]:
        run_script(name, cfg)
    simulate_ratings(cfg)
    for name in ["09_analyze_human", "10_figures"]:
        run_script(name, cfg)
    print("\n" + "=" * 70 + "\nSMOKE RUN FINISHED. Numbers are NOT meaningful (random networks, synthetic"
          f" images).\nOutputs: {out}\n" + "=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke_run")
    main(ap.parse_args().out)
