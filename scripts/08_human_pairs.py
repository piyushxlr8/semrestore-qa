"""08 - Build the blinded pairwise human study from HELD-OUT test images only.

Output: results/human/pairs.csv   (which images are in each pair; NOT shown to raters)
        results/human/rating.html (send this one file to raters; it works offline)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa.human import build_pairs, make_rating_html  # noqa: E402
from sqa.scoring import load_freeze  # noqa: E402
from sqa.utils import data_dir, ensure, load_config, results_dir  # noqa: E402


def main(cfg):
    load_freeze(cfg, data_dir(cfg) / "images.csv")       # metric must be frozen before the study
    res = results_dir(cfg)
    items = pd.read_csv(res / "items.csv")
    test = items[items.split == "test"]
    assert (test.split == "test").all()
    pairs = build_pairs(test, cfg, seed=cfg["seed"])
    out = ensure(res / "human")
    ensure(out / "ratings")
    pairs.to_csv(out / "pairs.csv", index=False)
    html = make_rating_html(pairs, cfg)
    (out / "rating.html").write_text(html)
    print(pairs.pair_type.value_counts().to_string())
    print(f"\n{len(pairs)} pairs -> {out / 'rating.html'} ({len(html) / 1e6:.1f} MB)")
    print("Send rating.html to raters. Put the CSV files they send back in results/human/ratings/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
