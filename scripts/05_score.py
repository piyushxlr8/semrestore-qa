"""05 - Score every (reference, test) pair with every metric. The only GPU-heavy metric step.

Output: results/items.csv     (what every scored item is)
        results/scores.csv    (raw metric values per item)
        results/maps/*.npz    (SHAFE-ResNet map + DINOv2 per-layer patch-distance maps)
Safe to re-run: items already scored are skipped.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa.items import build_items  # noqa: E402
from sqa.metrics import RawScorer  # noqa: E402
from sqa.utils import data_dir, ensure, get_device, load_config, load_image, results_dir, seed_everything  # noqa: E402


def main(cfg):
    seed_everything(cfg["seed"])
    res = ensure(results_dir(cfg))
    maps_dir = ensure(res / "maps")
    items = build_items(data_dir(cfg))
    items.to_csv(res / "items.csv", index=False)
    print("items to score:\n", items.groupby(["split", "kind"]).size())

    score_path = res / "scores.csv"
    old = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame(columns=["item_id"])
    todo = items[~items.item_id.isin(set(old.item_id))]
    print(f"\n{len(todo)} new items to score ({len(old)} already done)")
    if todo.empty:
        return

    scorer = RawScorer(cfg, get_device())
    bs = cfg["metrics"]["batch_size"]
    new_rows, t0 = [], time.time()
    for start in range(0, len(todo), bs):
        chunk = todo.iloc[start:start + bs]
        refs = [load_image(p) for p in chunk.ref_path]
        tests = [load_image(p) for p in chunk.test_path]
        rows, maps = scorer.score_batch(refs, tests)
        for item_id, row, m in zip(chunk.item_id, rows, maps):
            np.savez_compressed(maps_dir / f"{item_id}.npz", **m)
            new_rows.append({"item_id": item_id, **row})
        if start == 0:
            print(f"first batch: {time.time() - t0:.1f}s for {len(chunk)} items")
        if (start // bs) % 50 == 0 or start + bs >= len(todo):
            pd.concat([old, pd.DataFrame(new_rows)]).to_csv(score_path, index=False)
            print(f"  [{min(start + bs, len(todo))}/{len(todo)}]")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {score_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
