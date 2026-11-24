"""Are the top models really different? Paired day-block bootstrap of the MAE difference on a frozen split.

For each horizon the reference model (default LightGBM) is compared with every other model on the SAME rows and the
SAME resampled days, so differences that come from a lucky set of days cancel out. Reports the MAE difference
(other minus reference; positive = reference is better), its 95% interval, and the share of resamples in which the
reference wins. Descriptive only: it reads saved predictions and changes no model, setting or frozen file.

  python scripts/paired_bootstrap.py                  # test split, all rows and clean rows
  python scripts/paired_bootstrap.py --split val
Writes results/final/paired_{split}_h{h}.csv"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev          # noqa: E402
from src import models as mo            # noqa: E402


def paired(days, y, pa, pb, n_boot=1000, seed=0):
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    codes, _ = pd.factorize(np.asarray(days))
    D = int(codes.max()) + 1
    n = np.bincount(codes, minlength=D).astype("float64")
    sa = np.bincount(codes, weights=ea, minlength=D)
    sb = np.bincount(codes, weights=eb, minlength=D)
    W = np.random.default_rng(seed).multinomial(D, np.full(D, 1.0 / D), size=n_boot).astype("float64")
    N = W @ n
    diff = (W @ sb - W @ sa) / N                       # MAE(other) - MAE(reference), per resample
    return float((sb.sum() - sa.sum()) / n.sum()), np.percentile(diff, [2.5, 97.5]), float((diff > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--ref", default="lgbm")
    a = ap.parse_args()
    last = "2018-12-31 23:45" if a.split == "val" else None
    for h in ev.HORIZONS:
        names = ev.available(a.split, h)
        ref = ev.load_preds(a.split, a.ref, h)
        mask = ev.clean_mask(ref, last=last)
        P = {n: ev.load_preds(a.split, n, h)["pred"].to_numpy("float64") for n in names}
        y = ref["y_raw"].to_numpy("float64")
        days = ref["target_time"].dt.normalize().to_numpy()
        rows = []
        for view, m in (("all rows", np.ones(len(ref), bool)), ("clean rows", np.asarray(mask, bool))):
            for n in names:
                if n == a.ref:
                    continue
                d, (lo, hi), win = paired(days[m], y[m], P[a.ref][m], P[n][m])
                rows.append(dict(view=view, h=h, reference=mo.LABEL[a.ref], other=mo.LABEL[n],
                                 MAE_diff=d, lo=lo, hi=hi, ref_wins_share=win,
                                 verdict="reference better" if lo > 0 else ("other better" if hi < 0 else "not distinguishable")))
        out = pd.DataFrame(rows)
        out.to_csv(ev.FINAL / f"paired_{a.split}_h{h}.csv", index=False)
        print(f"\n--- {a.split}, {h} min: MAE(other) - MAE({mo.LABEL[a.ref]}), 95% day-block interval")
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
