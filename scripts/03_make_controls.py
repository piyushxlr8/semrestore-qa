"""03 - Faithful controls with the SAME PSNR as each edit, plus a fixed grid of degraded refs.

Output: data/controls/controls.csv  (one blur + one noise control per valid edit)
        data/degraded/degraded.csv  (every reference at fixed blur / noise levels)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa.controls import match_blur, match_noise  # noqa: E402
from sqa.utils import (add_noise, data_dir, ensure, gaussian_blur, load_config, load_image,  # noqa: E402
                       psnr, quantize, save_image)


def main(cfg):
    cc = cfg["controls"]
    root = data_dir(cfg)
    images = pd.read_csv(root / "images.csv").set_index("image_id")
    edits = pd.read_csv(root / "edits" / "edits.csv")
    edits = edits[edits.valid == 1]
    out = ensure(root / "controls")
    rows, dropped = [], 0
    for k, e in enumerate(edits.itertuples()):
        ref = load_image(images.loc[e.image_id, "ref_path"])
        target = psnr(load_image(e.edited_path), ref)
        for ctype in cc["types"]:
            if ctype == "blur":
                p, img, gap = match_blur(ref, target, cc["blur_sigma_range"], cc["psnr_tol_db"])
            else:
                p, img, gap = match_noise(ref, target, cc["noise_std_range"], cc["psnr_tol_db"], seed=k)
            if gap > cc["psnr_tol_db"]:
                dropped += 1
                continue
            cid = f"{e.edit_id}_ctrl{ctype}"
            path = out / "images" / f"{cid}.png"
            save_image(img, path)
            rows.append(dict(control_id=cid, edit_id=e.edit_id, control_type=ctype, param=round(p, 5),
                             control_path=str(path), psnr_edit=round(target, 3),
                             psnr_control=round(psnr(img, ref), 3), psnr_gap=round(gap, 4)))
        if (k + 1) % 50 == 0:
            print(f"[{k + 1}/{len(edits)}]")
    pd.DataFrame(rows).to_csv(out / "controls.csv", index=False)
    print(f"controls: {len(rows)} written, {dropped} could not be matched within "
          f"{cc['psnr_tol_db']} dB (dropped)")

    # fixed degradation grid for every reference
    dout = ensure(root / "degraded")
    drows = []
    for img_id, r in images.iterrows():
        ref = load_image(r.ref_path)
        grid = [("blur", s) for s in cc["degrade_blur_sigmas"]] + [("noise", s) for s in cc["degrade_noise_stds"]]
        for kind, p in grid:
            img = quantize(gaussian_blur(ref, p) if kind == "blur" else add_noise(ref, p, seed=0))
            did = f"{img_id}_{kind}{p}"
            path = dout / "images" / f"{did}.png"
            save_image(img, path)
            drows.append(dict(degraded_id=did, image_id=img_id, split=r.split, degradation=kind, param=p,
                              path=str(path)))
    pd.DataFrame(drows).to_csv(dout / "degraded.csv", index=False)
    print(f"degraded references: {len(drows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
