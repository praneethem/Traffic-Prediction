"""All nine models behind one interface: tune on validation, refit on train+validation, log cost.

Protocol (identical for every model and horizon)
  1. Target is the SCALED flow y = flow / sensor scale; predictions are multiplied back by sc (raw units).
  2. TUNE: random/grid search on a 300,000-row month-stratified subset of TRAIN, scored on the full validation
     split with raw-unit MAE. Boosting models also early-stop on validation (weighted by sc so the stopping
     metric equals raw-unit MAE).
  3. FIT the best configuration on ALL train rows -> validation predictions + best tree count are saved.
  4. REFIT (scripts/run_model.py refit): train + validation, early stopping OFF, tree count frozen at step 3,
     predict 2019 once. 2019 is never used in steps 1-3.
Set the environment variable TP_SMOKE=1 for a tiny, fast dry run of everything."""
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import evaluate as ev
from src import metrics as M
from src.features import ALL_FEATURES

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")        # LightGBM 4.7 wording; older versions need eval_set
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")             # Ridge on strongly correlated lags: harmless
SMOKE = os.environ.get("TP_SMOKE") == "1"
SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 1)            # keep one core free for the laptop
TUNE_ROWS = 20_000 if SMOKE else 300_000
FIT_CAP = 40_000 if SMOKE else None                    # cap rows of the "full" fits in smoke mode
STOP = 100                                             # early-stopping patience for boosting models

MODELS = ["ridge", "pca_ridge", "dt", "rf", "ada", "gb", "xgb", "lgbm", "cat"]
LABEL = {"ridge": "Ridge", "pca_ridge": "PCA + Ridge", "dt": "Decision Tree", "rf": "Random Forest",
         "ada": "AdaBoost", "gb": "Gradient Boosting", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost"}
BOOSTERS_ES = ("xgb", "lgbm", "cat")                   # models that early-stop on validation

# (search space, number of random draws; None = full grid)
SPACES = {
    "ridge": ({"alpha": [0.01, 0.1, 1, 10, 100, 1000]}, None),
    "pca_ridge": ({"k": [5, 10, 20, 30, len(ALL_FEATURES)]}, None),
    "dt": ({"max_depth": [6, 10, 14, 18], "min_samples_leaf": [20, 50, 100]}, None),
    "rf": ({"min_samples_leaf": [20, 40], "max_features": [0.33, 0.5, 0.7]}, 4),
    "ada": ({"max_depth": [3, 5], "learning_rate": [0.5, 1.0]}, None),
    "gb": ({"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}, None),
    "xgb": ({"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "min_child_weight": [1, 10, 50],
             "subsample": [0.7, 1.0], "colsample_bytree": [0.7, 1.0], "reg_lambda": [1, 10]}, 6),
    "lgbm": ({"num_leaves": [31, 63, 127, 255], "learning_rate": [0.05, 0.1], "min_child_samples": [20, 100, 300],
              "feature_fraction": [0.7, 1.0], "reg_lambda": [0, 10]}, 8),
    "cat": ({"depth": [6, 8], "learning_rate": [0.05, 0.1], "l2_leaf_reg": [3, 10]}, 4),
}
N_TREES = {"rf": 150, "ada": 100, "gb": 300}           # fixed tree counts for the bagging / classic boosting models
if SMOKE:
    N_TREES = {"rf": 10, "ada": 8, "gb": 25}
MAX_ITER = {"xgb": 3000, "lgbm": 3000, "cat": 2000}
if SMOKE:
    MAX_ITER = {"xgb": 60, "lgbm": 60, "cat": 60}


def _n_iter(name):
    space, n = SPACES[name]
    if n is None:
        return None
    return min(n, 2) if SMOKE else n


def candidates(name, seed=SEED):
    space, n = SPACES[name]
    grid = list(ParameterGrid(space))
    if _n_iter(name) is None:
        return grid if not SMOKE else grid[:2]
    return list(ParameterSampler(space, n_iter=min(_n_iter(name), len(grid)), random_state=seed))


# ----------------------------------------------------------------------------------------------- data helpers
def month_stratified_subset(df, n, seed=SEED):
    """Random subset that keeps the month mix of df (so the tuning subset is not dominated by one season)."""
    if n is None or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    month = df["time"].dt.to_period("M").astype(str).to_numpy()
    frac = n / len(df)
    keep = []
    for m in np.unique(month):
        idx = np.flatnonzero(month == m)
        keep.append(rng.choice(idx, size=max(1, int(round(len(idx) * frac))), replace=False))
    return df.iloc[np.sort(np.concatenate(keep))]


def cap(df, n=FIT_CAP, seed=SEED):
    if n is None or len(df) <= n:
        return df
    return df.iloc[np.sort(np.random.default_rng(seed).choice(len(df), size=n, replace=False))]


def xy(df):
    return df[ALL_FEATURES].to_numpy("float32"), df["y"].to_numpy("float32"), df["sc"].to_numpy("float64")


# ------------------------------------------------------------------------------------------------ model factory
def make_model(name, params, n_estimators=None, early_stop=False, seed=SEED, n_jobs=N_JOBS):
    p = dict(params)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if name == "pca_ridge":
        return make_pipeline(StandardScaler(), PCA(n_components=p["k"], random_state=seed), Ridge(alpha=p.get("alpha", 1.0)))
    if name == "dt":
        return DecisionTreeRegressor(random_state=seed, **p)
    if name == "rf":
        return RandomForestRegressor(n_estimators=n_estimators or N_TREES["rf"], max_samples=0.3,
                                     n_jobs=n_jobs, random_state=seed, **p)
    if name == "ada":
        depth = p.pop("max_depth")
        base = DecisionTreeRegressor(max_depth=depth, random_state=seed)
        try:
            return AdaBoostRegressor(estimator=base, n_estimators=n_estimators or N_TREES["ada"], loss="linear",
                                     random_state=seed, **p)
        except TypeError:                                  # older scikit-learn
            return AdaBoostRegressor(base_estimator=base, n_estimators=n_estimators or N_TREES["ada"], loss="linear",
                                     random_state=seed, **p)
    if name == "gb":
        return GradientBoostingRegressor(n_estimators=n_estimators or N_TREES["gb"], subsample=0.5, max_features=0.5,
                                         min_samples_leaf=20, random_state=seed, **p)
    if name == "xgb":
        import xgboost as xgb
        kw = dict(tree_method="hist", n_estimators=n_estimators or MAX_ITER["xgb"], n_jobs=n_jobs,
                  random_state=seed, eval_metric="mae", **p)
        if early_stop:
            kw["early_stopping_rounds"] = STOP
        return xgb.XGBRegressor(**kw)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="regression", n_estimators=n_estimators or MAX_ITER["lgbm"],
                                 n_jobs=n_jobs, random_state=seed, verbose=-1, **p)
    if name == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function="RMSE", eval_metric="MAE", iterations=n_estimators or MAX_ITER["cat"],
                                 thread_count=n_jobs, random_seed=seed, verbose=0, allow_writing_files=False, **p)
    raise ValueError(name)


def fit_model(name, model, Xtr, ytr, Xva=None, yva=None, wva=None, early_stop=False):
    """Fit; returns (model, best_iter or None, val_pred_scaled or None). wva = validation scale (sc) used as weights
    so that early stopping minimises raw-unit MAE."""
    if name == "lgbm":
        import lightgbm as lgb
        if early_stop:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_sample_weight=[wva], eval_metric="l1",
                      callbacks=[lgb.early_stopping(STOP, verbose=False)])
            return model, int(model.best_iteration_ or model.n_estimators), None
        model.fit(Xtr, ytr)
        return model, None, None
    if name == "xgb":
        if early_stop:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
            return model, int(model.best_iteration) + 1, None
        model.fit(Xtr, ytr, verbose=False)
        return model, None, None
    if name == "cat":
        if early_stop:
            from catboost import Pool
            model.fit(Xtr, ytr, eval_set=Pool(Xva, yva, weight=wva), early_stopping_rounds=STOP,
                      use_best_model=True, verbose=False)
            return model, int(model.get_best_iteration()) + 1, None
        model.fit(Xtr, ytr)
        return model, None, None
    if name == "gb" and early_stop:                        # pick the tree count with staged predictions
        model.fit(Xtr, ytr)
        best, best_i, best_p = np.inf, 0, None
        for i, p in enumerate(model.staged_predict(Xva)):
            e = float(np.mean(wva * np.abs(p - yva)))
            if e < best:
                best, best_i, best_p = e, i + 1, p
        return model, best_i, best_p
    model.fit(Xtr, ytr)
    return model, None, None


def val_mae_raw(y_scaled, p_scaled, sc):
    return float(np.mean(sc * np.abs(p_scaled - y_scaled)))


# ----------------------------------------------------------------------------------------------------- tuning
def tune_model(name, h, redo=False, verbose=True):
    """Full tuning + full-train fit for one model and horizon. Writes results/best/{name}_h{h}.json,
    results/logs/tuning_{name}_h{h}.csv and results/preds/val_{name}_h{h}.parquet. Safe to re-run (skips if done)."""
    out = ev.BEST / f"{name}_h{h}.json"
    if out.exists() and not redo:
        if verbose:
            print(f"[{name} h={h}] already done -> {out.name}")
        return json.loads(out.read_text())
    tr = ev.load_table(h, ["train"])
    va = ev.load_table(h, ["val"])
    sub = month_stratified_subset(tr, TUNE_ROWS)
    Xs, ys, _ = xy(sub)
    Xv, yv, scv = xy(va)
    early = name in BOOSTERS_ES or name == "gb"
    log, t_all = [], time.time()
    cfgs = candidates(name)
    if name == "pca_ridge":                                # reuse the tuned Ridge penalty when available
        r = ev.read_best("ridge", h)
        if r:
            cfgs = [{**c, "alpha": r["params"]["alpha"]} for c in cfgs]
    for i, c in enumerate(cfgs):
        t0 = time.time()
        m = make_model(name, c, early_stop=early)
        m, best_it, p_gb = fit_model(name, m, Xs, ys, Xv, yv, scv, early_stop=early)
        pv = p_gb if p_gb is not None else m.predict(Xv)
        mae = val_mae_raw(yv, pv, scv)
        log.append({**c, "best_iter": best_it, "val_MAE": mae, "fit_s": time.time() - t0})
        if verbose:
            print(f"[{name} h={h}] cfg {i + 1}/{len(cfgs)} {c} -> val MAE {mae:.2f}  ({time.time() - t0:.0f}s)")
    lg = pd.DataFrame(log)
    lg.to_csv(ev.LOGS / f"tuning_{name}_h{h}.csv", index=False)
    best_cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfgs[int(lg["val_MAE"].idxmin())].items()}

    full = cap(tr)
    Xf, yf, _ = xy(full)
    t0 = time.time()
    m = make_model(name, best_cfg, early_stop=early)
    m, best_it, p_gb = fit_model(name, m, Xf, yf, Xv, yv, scv, early_stop=early)
    fit_s = time.time() - t0
    t0 = time.time()
    pv = p_gb if p_gb is not None else m.predict(Xv)
    pred_s = time.time() - t0
    pred_raw = pv.astype("float64") * scv
    ev.save_preds("val", name, h, va.index.to_numpy(), pred_raw)
    res = {"model": name, "h": h, "params": best_cfg, "best_iter": best_it, "val": M.score(va["y_raw"], pred_raw),
           "fit_s_train": fit_s, "predict_s_per_100k": pred_s / len(va) * 1e5, "tune_total_s": time.time() - t_all,
           "tune_rows": len(sub), "fit_rows": len(full), "smoke": SMOKE}
    out.write_text(json.dumps(res, indent=2))
    if name == "lgbm":                                     # kept for the feature-importance panel and the app
        import joblib
        joblib.dump(m, ev.MODELS_DIR / f"lgbm_h{h}_trainonly.joblib")
    if verbose:
        print(f"[{name} h={h}] BEST {best_cfg} best_iter={best_it}  val MAE {res['val']['MAE']:.2f}  "
              f"(full fit {fit_s:.0f}s, total {res['tune_total_s'] / 60:.1f} min)")
    return res


# ----------------------------------------------------------------------------------------------------- refit
def refit_test(name, h, seed=SEED, tag="", force=False, verbose=True):
    """Refit on train+validation with the tree count frozen, predict 2019 ONCE, log the cost.
    Refuses to overwrite an existing 2019 prediction file (the test year is scored once)."""
    pth = ev.pred_path("test", name, h, tag)
    if pth.exists() and not force:
        if verbose:
            print(f"[{name} h={h}{tag}] 2019 predictions already exist -> skipped (test year is scored once)")
        return None
    best = ev.read_best(name, h)
    if best is None:
        raise RuntimeError(f"run the tuning for {name} h={h} first")
    trva = cap(ev.load_table(h, ["train", "val"]))
    te = ev.load_table(h, ["test"])
    Xf, yf, _ = xy(trva)
    Xt, _, sct = xy(te)
    n_fixed = best["best_iter"]
    m = make_model(name, best["params"], n_estimators=n_fixed, early_stop=False, seed=seed)
    t0 = time.time()
    m.fit(Xf, yf) if name not in ("xgb",) else m.fit(Xf, yf, verbose=False)
    fit_s = time.time() - t0
    t0 = time.time()
    pt = m.predict(Xt)
    pred_s = time.time() - t0
    ev.save_preds("test", name, h, te.index.to_numpy(), pt.astype("float64") * sct, tag=tag)
    cost = {"model": name, "h": h, "seed": seed, "fit_rows": len(trva), "n_estimators": n_fixed,
            "fit_s": fit_s, "predict_s_per_100k": pred_s / len(te) * 1e5,
            "model_MB": len(pickle.dumps(m)) / 1e6, "smoke": SMOKE}
    (ev.FINAL / f"cost_{name}_h{h}{tag}.json").write_text(json.dumps(cost, indent=2))
    if name == "lgbm" and tag == "":
        import joblib
        joblib.dump(m, ev.MODELS_DIR / f"lgbm_h{h}_final.joblib")
    if verbose:
        print(f"[{name} h={h}{tag}] refit {fit_s:.0f}s on {len(trva):,} rows, predicted {len(te):,} test rows -> {pth.name}")
    return cost


# ------------------------------------------------------------------------------------------- planning (Gate G1)
DEFAULT_PARAMS = {
    "ridge": {"alpha": 10}, "pca_ridge": {"k": 20, "alpha": 10}, "dt": {"max_depth": 12, "min_samples_leaf": 50},
    "rf": {"min_samples_leaf": 20, "max_features": 0.5}, "ada": {"max_depth": 5, "learning_rate": 1.0},
    "gb": {"max_depth": 4, "learning_rate": 0.1},
    "xgb": {"max_depth": 6, "learning_rate": 0.1, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
    "lgbm": {"num_leaves": 63, "learning_rate": 0.1, "min_child_samples": 100, "feature_fraction": 0.8},
    "cat": {"depth": 6, "learning_rate": 0.1},
}
EXPECTED_TREES = {"xgb": 600, "lgbm": 800, "cat": 800}   # rough guess used only for the time projection


def time_probe(name, n=40_000, probe_trees=None, seed=SEED):
    """Time a small fit and extrapolate cost per (tree x row). Returns seconds per tree-row unit and trees used."""
    tr = month_stratified_subset(ev.load_table(15, ["train"]), n, seed)
    X, y, _ = xy(tr)
    probe_trees = probe_trees or {"rf": 20, "ada": 10, "gb": 20, "xgb": 100, "lgbm": 100, "cat": 100}.get(name)
    m = make_model(name, DEFAULT_PARAMS[name], n_estimators=probe_trees)
    t0 = time.time()
    m.fit(X, y)
    t = time.time() - t0
    return {"model": name, "probe_rows": len(tr), "probe_trees": probe_trees, "probe_s": t}


def project_hours(name, probe):
    """Rough total hours for the full protocol (both horizons): tuning configs on the tuning subset, one full-train
    fit, one train+validation refit. Cost model: time ~ trees x rows^1.15."""
    trees_plan = N_TREES.get(name) or EXPECTED_TREES.get(name)
    ptrees = probe["probe_trees"] or 1
    k = probe["probe_s"] / (ptrees * probe["probe_rows"] ** 1.15)

    def t(rows, trees):
        return k * (trees or 1) * rows ** 1.15
    n_cfg = len(candidates(name)) if not SMOKE else 2
    tune_t = n_cfg * t(300_000, trees_plan)
    full_t = t(993_000, trees_plan)
    refit_t = t(1_343_000, trees_plan) * (3 if name == "lgbm" else 1)
    return 2 * (tune_t + full_t + refit_t) / 3600
