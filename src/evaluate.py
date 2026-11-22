"""Shared harness: loads the feature tables, builds baselines, saves/loads predictions, scores every model the same way.

Predictions are always stored in RAW units (vehicles per 15 min) next to the row number of the feature
table, so any model can be re-scored later without refitting."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

from src.features import ALL_FEATURES, FEATURE_GROUPS          # noqa: F401  (re-exported)
from src import metrics as M
from src import quality as Q

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SMOKE = os.environ.get("TP_SMOKE") == "1"      # dry runs write to results_smoke/ so they can never block real runs
RESULTS = ROOT / ("results_smoke" if SMOKE else "results")
FIGS = RESULTS / "figures"
PREDS = RESULTS / "preds"
BEST = RESULTS / "best"
LOGS = RESULTS / "logs"
FINAL = RESULTS / "final"
MODELS_DIR = ROOT / ("models_smoke" if SMOKE else "models")
for _d in (RESULTS, FIGS, PREDS, BEST, LOGS, FINAL, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASELINES = {"Persistence": "flow_t", "Last week": "same_slot_last_week", "4-week average": "same_slot_4wk_mean"}
HORIZONS = (15, 60)


def load_table(h, splits=None, columns=None):
    """Feature table for horizon h (minutes). The row index is the row number in the parquet file."""
    df = pd.read_parquet(DATA / f"features_h{h}.parquet", columns=columns)
    for c in ("time", "target_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    if splits is not None:
        df = df[df["split"].isin(list(splits))]
    return df


def clean_mask(df, last=None):
    """Rows whose target day is not a repeated-template day (see src/quality.py). Pass last='2018-12-31 23:45' in
    notebooks that must not open 2019."""
    F = pd.read_parquet(DATA / "flow_15min_100.parquet")
    if last is not None:
        F = F.loc[:last]
    return Q.row_mask(df, Q.template_days(F))


def baseline_preds(df):
    """Raw-unit predictions of the three simple baselines."""
    return {name: (df[col] * df["sc"]).to_numpy("float64") for name, col in BASELINES.items()}


def baseline_table(df_val, mask=None):
    if mask is not None:
        df_val = df_val[mask]
    rows = []
    for name, p in baseline_preds(df_val).items():
        rows.append({"Model": name, **M.score(df_val["y_raw"], p)})
    return pd.DataFrame(rows).set_index("Model")


def best_baseline(df_val):
    """Name of the baseline with the lowest validation MAE (chosen on validation, never on 2019)."""
    t = baseline_table(df_val)
    return t["MAE"].idxmin()


def pred_path(split, name, h, tag=""):
    return PREDS / f"{split}_{name}_h{h}{tag}.parquet"


def save_preds(split, name, h, row_index, pred_raw, tag=""):
    pd.DataFrame({"row": np.asarray(row_index, "int64"),
                  "pred": np.asarray(pred_raw, "float32")}).to_parquet(pred_path(split, name, h, tag))


def load_preds(split, name, h, tag=""):
    """Returns the feature-table rows of `split` with an added column `pred` (raw units)."""
    p = pd.read_parquet(pred_path(split, name, h, tag))
    df = load_table(h).loc[p["row"].to_numpy()].copy()
    df["pred"] = p["pred"].to_numpy("float64")
    return df


def available(split, h):
    """Model names that have a saved prediction file for this split and horizon (seed variants excluded)."""
    out = []
    for f in sorted(PREDS.glob(f"{split}_*_h{h}.parquet")):
        out.append(f.name[len(split) + 1: -len(f"_h{h}.parquet")])
    return out


def read_best(name, h):
    f = BEST / f"{name}_h{h}.json"
    return json.loads(f.read_text()) if f.exists() else None


def leaderboard(split, h, names=None, clean=False):
    """One row per model: MAE, RMSE, WAPE and skill vs the best baseline (best baseline chosen on VALIDATION)."""
    val = load_table(h, ["val"])
    bb = best_baseline(val)          # always chosen on all validation rows, never on 2019
    rows = []
    for n in (names or available(split, h)):
        d = load_preds(split, n, h)
        if clean:
            d = d[clean_mask(d, last="2018-12-31 23:45" if split != "test" else None)]
        base = (d[BASELINES[bb]] * d["sc"]).to_numpy("float64")
        s = M.score(d["y_raw"], d["pred"])
        s["Skill"] = M.skill(s["MAE"], M.mae(d["y_raw"], base))
        rows.append({"Model": n, **s})
    out = pd.DataFrame(rows).set_index("Model").sort_values("MAE")
    out.attrs["best_baseline"] = bb
    return out
