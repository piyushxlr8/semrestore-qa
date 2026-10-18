"""06 - Calibrate and tune on DEV ONLY, then freeze the metric.

1. Calibration: per-layer mean/std of DINOv2 patch distances on dev references that are only
   blurred or noised (no content change). "This much distance is normal for blur."
2. Tuning: for each DINO-based variant, try the small grid in config.yaml and keep the setting
   with the best mean dev AUROC (edits vs PSNR-matched controls).
3. Thresholds: for every metric, the F1-best threshold and the 5%-FPR threshold on dev.
4. Write results/freeze.json with a fingerprint of the metric code + config + split.
   07/08 refuse to run on the held-out test split if anything changed afterwards.
"""
import argparse
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa import metrics as M  # noqa: E402
from sqa.evaluation import best_f1_threshold, fpr_threshold  # noqa: E402
from sqa.scoring import (badness_table, detection_split, dino_scores, load_maps,  # noqa: E402
                         mean_auroc_over_types, metric_columns, metrics_fingerprint)
from sqa.utils import data_dir, ensure, load_config, results_dir, write_json  # noqa: E402


def main(cfg):
    res = results_dir(cfg)
    tc = cfg["tuning"]
    items = pd.read_csv(res / "items.csv")
    scores = pd.read_csv(res / "scores.csv")
    dev = items[items.split == "dev"]                       # <- the ONLY data this script uses
    assert (dev.split == "dev").all()
    stored = cfg["metrics"]["dino"]["layers"]

    calib_ids = dev[dev.kind == "degraded"].item_id
    stats = M.calibration_stats(list(load_maps(res, calib_ids).values()))
    print(f"calibration: {len(calib_ids)} blurred/noised dev references")

    eval_items = dev[dev.kind.isin(["edit", "control_blur", "control_noise"])]
    maps = load_maps(res, eval_items.item_id)
    base = badness_table(eval_items, scores, {})

    grid_rows, best = [], {}
    layer_sets = [ls for ls in tc["layer_subsets"] if all(l in stored for l in ls)]
    for variant, (calibrated, pooling) in M.DINO_VARIANTS.items():
        if pooling == "topk":
            knob, values = "k_frac", tc["topk_fracs"]
        else:
            knob, values = "tau", tc["calibrated_temperatures"] if calibrated else tc["softmax_temperatures"]
        for layers, v in itertools.product(layer_sets, values):
            params = {"layers": layers, knob: v}
            s = dino_scores(maps, variant, params, stored, stats)
            bad = base.copy()
            bad[variant] = M.to_badness(variant, bad.item_id.map(s))
            obj = mean_auroc_over_types(bad, variant)
            grid_rows.append(dict(variant=variant, layers=str(layers), knob=knob, value=v, dev_mean_auroc=obj))
            if variant not in best or obj > best[variant][1]:
                best[variant] = (params, obj)
        print(f"{variant:18s} best on dev: {best[variant][0]}  mean AUROC={best[variant][1]:.3f}")

    params = {v: p for v, (p, _) in best.items()}
    derived = {v: dino_scores(maps, v, params[v], stored, stats) for v in params}
    bad = badness_table(eval_items, scores, derived)

    thresholds = {}
    for m in metric_columns(bad):
        e, c = detection_split(bad, m)
        thresholds[m] = {"f1": best_f1_threshold(e[m], c[m]),
                         f"fpr_{tc['fpr_target']}": fpr_threshold(c[m], tc["fpr_target"])}

    ensure(res / "tables")
    pd.DataFrame(grid_rows).to_csv(res / "tables" / "dev_tuning_grid.csv", index=False)
    freeze = dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        fingerprint=metrics_fingerprint(cfg, data_dir(cfg) / "images.csv"),
        stored_layers=stored, calibration=stats, params=params,
        dev_mean_auroc={v: o for v, (_, o) in best.items()},
        thresholds=thresholds, fpr_target=tc["fpr_target"],
        n_dev_items={k: int(n) for k, n in eval_items.kind.value_counts().items()},
    )
    write_json(freeze, res / "freeze.json")
    print(f"\nFrozen -> {res / 'freeze.json'}. Do not change sqa/metrics.py or the metrics/tuning "
          "config from here on (the test scripts will refuse).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
