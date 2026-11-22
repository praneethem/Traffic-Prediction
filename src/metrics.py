"""Error metrics (all in vehicles per 15 min) and a day-block bootstrap for confidence intervals."""
import numpy as np
import pandas as pd


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, "float64") - np.asarray(p, "float64"))))


def rmse(y, p):
    d = np.asarray(y, "float64") - np.asarray(p, "float64")
    return float(np.sqrt(np.mean(d * d)))


def wape(y, p):
    """Weighted absolute percentage error = sum|error| / sum|actual| (safe when actual is 0)."""
    y = np.asarray(y, "float64")
    return float(np.abs(y - np.asarray(p, "float64")).sum() / np.abs(y).sum())


def skill(mae_model, mae_baseline):
    """1 - MAE(model)/MAE(baseline). Positive = the model beats the baseline."""
    return 1.0 - mae_model / mae_baseline


def score(y, p):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE": wape(y, p), "n": int(len(y))}


def day_block_bootstrap(days, y, p, p_base=None, n_boot=1000, seed=0, level=0.95):
    """Resample whole DAYS with replacement (errors within a day are strongly correlated, so
    resampling single rows would give intervals that are far too narrow).
    Returns {metric: (low, high)} for MAE, RMSE, WAPE and, if p_base is given, skill."""
    y = np.asarray(y, "float64")
    e = np.abs(y - np.asarray(p, "float64"))
    codes, _ = pd.factorize(np.asarray(days))
    D = int(codes.max()) + 1
    n = np.bincount(codes, minlength=D).astype("float64")
    s_abs = np.bincount(codes, weights=e, minlength=D)
    s_sq = np.bincount(codes, weights=e * e, minlength=D)
    s_y = np.bincount(codes, weights=np.abs(y), minlength=D)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot).astype("float64")   # (B, D) day counts
    N = W @ n
    out = {"MAE": W @ s_abs / N, "RMSE": np.sqrt(W @ s_sq / N), "WAPE": (W @ s_abs) / (W @ s_y)}
    if p_base is not None:
        s_b = np.bincount(codes, weights=np.abs(y - np.asarray(p_base, "float64")), minlength=D)
        out["Skill"] = 1.0 - (W @ s_abs) / (W @ s_b)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in out.items()}
