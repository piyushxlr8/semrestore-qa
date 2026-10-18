"""04 - Real restoration (x4 super-resolution) with three model families, on held-out images.

For each test image we restore
  * the clean reference          -> "restored_ref"  (real restorer output; human study uses these)
  * up to 2 of its edited images -> "restored_edit" (edit-then-restore: compared to the CLEAN ref,
                                                     this is a restorer output with a known
                                                     content change)
Output: data/restored/restored.csv, data/restored/images/*.png
Safe to re-run: finished outputs are skipped.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa.restorers import degrade_sr_x4, load_restorer  # noqa: E402
from sqa.utils import data_dir, ensure, get_device, load_config, load_image, save_image  # noqa: E402

EDIT_PREFERENCE = ["substitution", "addition", "removal"]


def main(cfg, families=None):
    rc = cfg["restore"]
    root = data_dir(cfg)
    images = pd.read_csv(root / "images.csv")
    edits = pd.read_csv(root / "edits" / "edits.csv")
    edits = edits[edits.valid == 1]
    test = images[images.split == "test"].head(rc["n_images"])
    out = ensure(root / "restored")
    csv_path = out / "restored.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["restore_id"] for r in rows}

    # the jobs: (source id, image to degrade)
    jobs = []
    for r in test.itertuples():
        jobs.append((r.image_id, "ref", r.ref_path))
        mine = edits[edits.image_id == r.image_id]
        mine = mine[mine.edit_type.isin(EDIT_PREFERENCE)]
        mine = mine.assign(o=mine.edit_type.map(EDIT_PREFERENCE.index)).sort_values("o")
        for e in mine.head(rc["edits_per_image"]).itertuples():
            jobs.append((r.image_id, e.edit_id, e.edited_path))

    device = get_device()
    fake = bool(cfg["metrics"].get("fake_models", False))
    split_of = images.set_index("image_id").split
    for family in families or list(rc["families"]):
        todo = [j for j in jobs if f"{j[1] if j[1] != 'ref' else j[0]}_sr_{family}" not in done]
        if not todo:
            continue
        print(f"\n== {family}: {len(todo)} images")
        model = load_restorer(family, cfg, device, fake=fake)
        t0 = time.time()
        for k, (image_id, source, path) in enumerate(todo):
            rid = f"{source if source != 'ref' else image_id}_sr_{family}"
            lr = degrade_sr_x4(load_image(path))
            sr = model(lr, seed=cfg["seed"] + k)
            op = out / "images" / f"{rid}.png"
            save_image(sr, op)
            lp = out / "lr" / f"{source if source != 'ref' else image_id}_lr.png"
            if not lp.exists():
                save_image(lr, lp)
            rows.append(dict(restore_id=rid, image_id=image_id, split=split_of[image_id], task=rc["task"],
                             family=family, source=source, lr_path=str(lp), out_path=str(op)))
            if k == 0:
                print(f"  first image: {time.time() - t0:.1f}s")
            if (k + 1) % 20 == 0:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"  [{k + 1}/{len(todo)}]")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"\nrestored outputs: {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--families", nargs="*", default=None, help="e.g. --families regression gan")
    a = ap.parse_args()
    main(load_config(a.config), a.families)
