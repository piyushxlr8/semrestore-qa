"""01 - Pick 300 COCO images with one editable object each; split into dev / held-out test.

Output: data/images.csv, data/refs/*.png, data/masks/*.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from sqa.data import add_images, build_dataset  # noqa: E402
from sqa.utils import data_dir, load_config  # noqa: E402


def main(cfg, add=None, add_dev=None):
    out = data_dir(cfg) / "images.csv"
    if add:   # scale up: keep the existing images and their split, append new ones
        old = pd.read_csv(out)
        if "batch" in old and (old.batch == "new").any():
            raise SystemExit("images.csv already has added images; refusing to add twice.")
        df = add_images(cfg, old, add, add_dev)
        (data_dir(cfg) / "images_before_scaling.csv").write_text(old.to_csv(index=False))
    else:
        df = build_dataset(cfg)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}: {len(df)} images  ->  {df.split.value_counts().to_dict()}")
    if "batch" in df:
        print(df.groupby(["batch", "split"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--add", type=int, default=None, help="append this many NEW images to images.csv")
    ap.add_argument("--add-dev", type=int, default=None, help="how many of the new images go to dev")
    a = ap.parse_args()
    main(load_config(a.config), a.add, a.add_dev)
