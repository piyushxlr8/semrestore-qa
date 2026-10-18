"""09 - Analyse the human ratings (raw CSVs are only read, never modified).

Output (results/tables/):
  h1_raters.csv          gold-check accuracy per rater and who is kept (rule fixed in config)
  h2_agreement.csv       Krippendorff's alpha and split-half agreement (= human ceiling)
  h3_metric_vs_human.csv pairwise accuracy + Spearman / Pearson (metric margin vs vote share)
  h4_edit_validation.csv how often humans agree the edit changed content (edit_vs_blur pairs)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa.evaluation import bootstrap_ci  # noqa: E402
from sqa.human import alpha_for, metric_vs_human, rater_gold_accuracy, split_half_agreement, vote_table  # noqa: E402
from sqa.scoring import metric_columns  # noqa: E402
from sqa.utils import ensure, load_config, results_dir  # noqa: E402


def main(cfg, ratings_dir=None):
    res = results_dir(cfg)
    hdir = res / "human"
    rdir = Path(ratings_dir) if ratings_dir else hdir / "ratings"
    files = sorted(rdir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No rating CSVs found in {rdir}")
    ratings = pd.concat([pd.read_csv(f, dtype={"rater": str}) for f in files], ignore_index=True)
    ratings = ratings.drop_duplicates(["rater", "pair_id"], keep="last")
    pairs = pd.read_csv(hdir / "pairs.csv")
    tdir = ensure(res / "tables")

    # ---- rater quality control (exclusion rule fixed in config BEFORE collection)
    gold = rater_gold_accuracy(ratings, pairs)
    keep = gold[gold >= cfg["human"]["min_gold_accuracy"]].index
    h1 = pd.DataFrame({"gold_accuracy": gold, "n_answers": ratings.groupby("rater").size(),
                       "kept": gold.index.isin(keep)})
    h1.to_csv(tdir / "h1_raters.csv")
    print(h1, f"\nkept {len(keep)} of {len(gold)} raters")
    ratings = ratings[ratings.rater.isin(keep)]
    if ratings.empty:
        raise SystemExit("No raters passed the gold check.")

    # ---- agreement
    nongold = ratings[ratings.pair_id.isin(pairs[pairs.pair_type != "gold"].pair_id)]
    h2 = [dict(pair_type="all (non-gold)", n_raters=nongold.rater.nunique(),
               krippendorff_alpha=alpha_for(nongold), split_half_agreement=split_half_agreement(nongold))]
    for pt, ids in pairs.groupby("pair_type").pair_id:
        if pt == "gold":
            continue
        sub = nongold[nongold.pair_id.isin(ids)]
        if len(sub):
            h2.append(dict(pair_type=pt, n_raters=sub.rater.nunique(), krippendorff_alpha=alpha_for(sub),
                           split_half_agreement=split_half_agreement(sub)))
    h2 = pd.DataFrame(h2)
    h2.to_csv(tdir / "h2_agreement.csv", index=False)
    print("\n", h2.to_string(index=False))

    # ---- metric vs humans
    votes = vote_table(ratings)
    bad = pd.read_csv(res / "badness.csv").set_index("item_id")
    badness = {m: bad[m].to_dict() for m in metric_columns(bad.reset_index())}
    h3 = metric_vs_human(votes, pairs, badness)
    h3.to_csv(tdir / "h3_metric_vs_human.csv", index=False)
    print("\n", h3.round(3).to_string(index=False))

    # ---- do humans confirm the edits changed content? (validates the benchmark itself)
    v = votes.merge(pairs, on="pair_id")
    h4 = []
    for pt in ["edit_vs_blur", "severity"]:
        g = v[v.pair_type == pt].copy()
        if g.empty:
            continue
        # edit_vs_blur: A = edit, B = control  -> humans "should" pick B
        # severity:     A = small change      -> humans "should" pick A
        target = "b" if pt == "edit_vs_blur" else "a"
        g["hit"] = (g.majority == target).astype(float)
        lo, hi = bootstrap_ci(g.reset_index(drop=True), lambda d: d.hit.mean(), group_col="pair_id",
                              iters=cfg["evaluation"]["bootstrap_iters"])
        h4.append(dict(pair_type=pt, n_pairs=len(g), humans_pick_expected=g.hit.mean(), lo=lo, hi=hi,
                       humans_say_same=float(np.mean(g.majority == "same"))))
    h4 = pd.DataFrame(h4)
    h4.to_csv(tdir / "h4_edit_validation.csv", index=False)
    print("\n", h4.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ratings-dir", default=None)
    a = ap.parse_args()
    main(load_config(a.config), a.ratings_dir)
