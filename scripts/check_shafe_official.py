"""Check our SHAFE-ResNet against the official HalluGen SHAFE.py on real image pairs.

    git clone https://github.com/edshkim98/HalluGen.git ../HalluGen
    python scripts/check_shafe_official.py --official ../HalluGen/SHAFE.py

The official forward() is written for 1-channel MRI (it repeats the channel 3x), so for RGB we call
its own pieces in the same order: low-pass (radius 60) -> ImageNet norm -> ETHFeature (resize to
224, ResNet features at out_indices [1, 2], cosine map, softmax pooling with tau = 0.02).
Output: results/tables/shafe_official_check.csv
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa.metrics import RawScorer  # noqa: E402
from sqa.utils import ensure, load_config, load_image, results_dir  # noqa: E402


def main(cfg, official_path, n):
    import torch
    spec = importlib.util.spec_from_file_location("official_shafe", official_path)
    off = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(off)
    official = off.SHAFE(model_name=cfg["metrics"]["shafe_resnet"]["model"], device="cpu")

    # official radius 60 is for 256 px images; we compare at 256 px so both use exactly radius 60
    items = pd.read_csv(results_dir(cfg) / "items.csv")
    items = items[items.kind.isin(["edit", "control_blur", "control_noise"])].sample(n, random_state=0)
    ours = RawScorer(cfg, "cpu")
    rows = []
    for r in items.itertuples():
        ref = torch.from_numpy(load_image(r.ref_path).transpose(2, 0, 1))[None]
        test = torch.from_numpy(load_image(r.test_path).transpose(2, 0, 1))[None]
        ref = torch.nn.functional.interpolate(ref, size=(256, 256), mode="bilinear", antialias=True)
        test = torch.nn.functional.interpolate(test, size=(256, 256), mode="bilinear", antialias=True)
        with torch.no_grad():
            gt = official._apply_lowpass_batched(ref.clamp(0, 1), cutoff_radius=60)
            pr = official._apply_lowpass_batched(test.clamp(0, 1), cutoff_radius=60)
            gt, pr = (gt - official.mean) / official.std, (pr - official.mean) / official.std
            s_off = float(official.eth_feature(pr, gt)[0])
            s_ours = float(ours._shafe_resnet(ref, test)[0][0])
        rows.append(dict(item_id=r.item_id, kind=r.kind, official=s_off, ours=s_ours,
                         abs_diff=abs(s_off - s_ours)))
        print(f"{r.item_id:45s} official={s_off:.6f} ours={s_ours:.6f}")
    df = pd.DataFrame(rows)
    df.to_csv(ensure(results_dir(cfg) / "tables") / "shafe_official_check.csv", index=False)
    print(f"\nmax |difference| over {len(df)} pairs: {df.abs_diff.max():.2e}  "
          f"(Pearson r = {np.corrcoef(df.official, df.ours)[0, 1]:.6f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--official", required=True, help="path to HalluGen's SHAFE.py")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    main(load_config(a.config), a.official, a.n)
