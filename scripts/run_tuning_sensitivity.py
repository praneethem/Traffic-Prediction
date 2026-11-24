"""How much did the small tuning budget cost? (validation only; 2019 is never opened, nothing frozen is touched)

Several frozen best settings sit on the edge of the original search grid (see README, Limitations). This script runs a
SECOND, wider random search around them for LightGBM, XGBoost and CatBoost with exactly the same protocol as the first
(300,000-row month-stratified subset of train, early stopping on validation, scored on the full validation year), then
refits the best new setting on all train rows and compares it with the frozen one on validation.

  python scripts/run_tuning_sensitivity.py --h 15 60                 # all three models
  python scripts/run_tuning_sensitivity.py --h 15 --models lgbm
Writes results/tuning_sensitivity/{model}_h{h}.csv (every draw) and results/tuning_sensitivity/summary.csv.
It does NOT change results/best, models/ or FROZEN.json: it measures the headroom, it does not replace the frozen models."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev          # noqa: E402
from src import models as mo            # noqa: E402

OUT = ev.RESULTS / "tuning_sensitivity"
OUT.mkdir(exist_ok=True)

# Wider spaces. Each one continues PAST the edge the frozen setting sat on: more regularisation and larger leaves for
# LightGBM, deeper trees for XGBoost and CatBoost, a smaller step size for LightGBM, a higher iteration cap for CatBoost.
SPACES2 = {
    "lgbm": ({"num_leaves": [127, 255, 511], "min_child_samples": [300, 600, 1000], "reg_lambda": [10, 30, 100],
              "learning_rate": [0.03, 0.05], "feature_fraction": [0.85, 1.0]}, 8),
    "xgb": ({"max_depth": [8, 10, 12], "min_child_weight": [1, 3, 10], "reg_lambda": [1, 10],
             "learning_rate": [0.05], "subsample": [1.0], "colsample_bytree": [1.0]}, 6),
    "cat": ({"depth": [8, 10], "l2_leaf_reg": [3, 10, 30], "learning_rate": [0.1, 0.15]}, 4),
}
CAT_ITER = 4000                                       # the first search was capped at 2,000 and hit the cap


def run(name, h, draws=None):
    space, n = SPACES2[name]
    n = draws or n
    if name == "cat":
        mo.MAX_ITER["cat"] = CAT_ITER
    tr, va = ev.load_table(h, ["train"]), ev.load_table(h, ["val"])
    sub = mo.month_stratified_subset(tr, mo.TUNE_ROWS)
    Xs, ys, _ = mo.xy(sub)
    Xv, yv, scv = mo.xy(va)
    cfgs = list(ParameterSampler(space, n_iter=min(n, len(list(ParameterGrid(space)))), random_state=mo.SEED + 1))
    log = []
    for i, c in enumerate(cfgs):
        t0 = time.time()
        m = mo.make_model(name, c, early_stop=True)
        m, it, _ = mo.fit_model(name, m, Xs, ys, Xv, yv, scv, early_stop=True)
        mae = mo.val_mae_raw(yv, m.predict(Xv), scv)
        log.append({**c, "best_iter": it, "val_MAE_tune_subset": mae, "fit_s": time.time() - t0})
        print(f"[{name} h={h}] draw {i + 1}/{len(cfgs)} {c} -> {mae:.3f} ({time.time() - t0:.0f}s)", flush=True)
    lg = pd.DataFrame(log)
    lg.to_csv(OUT / f"{name}_h{h}.csv", index=False)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfgs[int(lg["val_MAE_tune_subset"].idxmin())].items()}
    full = mo.cap(tr)
    Xf, yf, _ = mo.xy(full)
    t0 = time.time()
    m = mo.make_model(name, best, early_stop=True)
    m, it, _ = mo.fit_model(name, m, Xf, yf, Xv, yv, scv, early_stop=True)
    new = mo.val_mae_raw(yv, m.predict(Xv), scv)
    frozen = ev.read_best(name, h)
    old = float(frozen["val"]["MAE"])
    row = dict(model=name, h=h, frozen_val_MAE=old, wider_search_val_MAE=new, change_pct=(new / old - 1) * 100,
               frozen_params=str(frozen["params"]), wider_params=str(best), frozen_iter=frozen["best_iter"], wider_iter=it,
               draws=len(cfgs), full_fit_s=time.time() - t0)
    print(f"[{name} h={h}] frozen {old:.3f} -> wider search {new:.3f} ({row['change_pct']:+.2f}%) {best} iter={it}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lgbm", "xgb", "cat"], choices=list(SPACES2))
    ap.add_argument("--h", nargs="+", type=int, default=[15, 60])
    ap.add_argument("--draws", type=int, default=None, help="override the number of random draws (testing)")
    a = ap.parse_args()
    f = OUT / "summary.csv"
    rows = []
    for name in a.models:
        for h in a.h:
            rows.append(run(name, h, a.draws))
            old = pd.read_csv(f) if f.exists() else pd.DataFrame()
            if not old.empty:
                old = old[~((old["model"] == name) & (old["h"] == h))]
            pd.concat([old, pd.DataFrame([rows[-1]])], ignore_index=True).to_csv(f, index=False)
    print(pd.read_csv(f)[["model", "h", "frozen_val_MAE", "wider_search_val_MAE", "change_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
