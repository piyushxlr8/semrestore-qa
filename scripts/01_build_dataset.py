"""01 - Pick 300 COCO images with one editable object each; split into dev / held-out test.

Output: data/images.csv, data/refs/*.png, data/masks/*.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa.data import build_dataset          # noqa: E402
from sqa.utils import data_dir, load_config  # noqa: E402


def main(cfg):
    df = build_dataset(cfg)
    out = data_dir(cfg) / "images.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}: {len(df)} images  ->  {df.split.value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
