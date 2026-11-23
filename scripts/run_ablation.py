"""Analysis A: feature-group ablation (validation only, 2019 is never opened).

For every model and horizon this script fits the model's FROZEN best configuration (results/best/*.json) on a
month-stratified subset of the TRAIN split and scores it on the full VALIDATION split in raw units.

  all            : every feature (the reference for that model and subset size)
  drop_<Group>   : all features except one group (7 groups -> 7 fits)
  cum_<k>        : LightGBM only, groups added one at a time in the order Lags, Seasonal, Calendar, Rolling,
                   Neighbours, Weather, Context (a "build-up" curve)

9 models x 8 configs = 72 fits per horizon (+ 6 build-up fits for LightGBM). Each model is compared with its OWN
reference at the same number of training rows, so row caps for the slow models do not bias the comparison.

  python scripts/run_ablation.py --h 15 60                      # all models that have a tuned config
  python scripts/run_ablation.py --h 15 60 --models lgbm rf     # a subset
  python scripts/run_ablation.py --plot-only                    # redraw the figures from the saved CSVs
Safe to stop and restart: finished (model, config) pairs are skipped.  --rows N overrides the row cap (testing)."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev          # noqa: E402
from src import metrics as M            # noqa: E402
from src import models as mo            # noqa: E402
from src.features import ALL_FEATURES, FEATURE_GROUPS   # noqa: E402

OUT = ev.RESULTS / "ablation"
OUT.mkdir(exist_ok=True)
ROWS = {"ada": 100_000, "gb": 150_000}          # training rows per fit (default below); slow models get fewer
DEFAULT_ROWS = 300_000
BUILD_ORDER = ["Lags", "Seasonal", "Calendar", "Rolling", "Neighbours", "Weather", "Context"]
COLS = ["model", "h", "config", "n_features", "train_rows", "MAE", "RMSE", "WAPE", "fit_s"]


def configs(name):
    out = [("all", ALL_FEATURES)]
    for g, fs in FEATURE_GROUPS.items():
        out.append((f"drop_{g}", [f for f in ALL_FEATURES if f not in fs]))
    if name == "lgbm":
        for k in range(1, len(BUILD_ORDER)):
            cols = [f for g in BUILD_ORDER[:k] for f in FEATURE_GROUPS[g]]
            out.append((f"cum_{k}_{BUILD_ORDER[k - 1]}", cols))
    return out


def csv_path(h):
    return OUT / f"ablation_h{h}.csv"


def load_done(h):
    p = csv_path(h)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=COLS)


def usable(name, best):
    if best is None:
        return "no tuned config yet"
    if name == "ada" and best["params"].get("max_depth", 0) < 8:
        return "AdaBoost config is from the OLD grid; wait for the re-tune to finish"
    return None


def run(models, horizons, rows_override, jobs):
    for h in horizons:
        tr = ev.load_table(h, ["train"])
        va = ev.load_table(h, ["val"])
        Xv_all, scv = va, va["sc"].to_numpy("float64")
        for name in models:
            best = ev.read_best(name, h)
            why = usable(name, best)
            if why:
                print(f"[{name} h={h}] skipped: {why}")
                continue
            rows = rows_override or ROWS.get(name, DEFAULT_ROWS)
            sub = mo.month_stratified_subset(tr, rows)
            y = sub["y"].to_numpy("float32")
            n_est = best.get("best_iter")
            for cfg, cols in configs(name):
                done = load_done(h)
                if ((done["model"] == name) & (done["config"] == cfg)).any():
                    continue
                p = dict(best["params"])
                if name == "pca_ridge":
                    p["k"] = min(p["k"], len(cols))
                t0 = time.time()
                m = mo.make_model(name, p, n_estimators=n_est, early_stop=False, n_jobs=jobs)
                m, _, _ = mo.fit_model(name, m, sub[cols].to_numpy("float32"), y)
                fit_s = time.time() - t0
                pv = m.predict(Xv_all[cols].to_numpy("float32")).astype("float64") * scv
                s = M.score(va["y_raw"], pv)
                row = {"model": name, "h": h, "config": cfg, "n_features": len(cols), "train_rows": len(sub),
                       "MAE": s["MAE"], "RMSE": s["RMSE"], "WAPE": s["WAPE"], "fit_s": fit_s}
                pd.concat([done, pd.DataFrame([row])], ignore_index=True).to_csv(csv_path(h), index=False)
                print(f"[{name} h={h}] {cfg:<22} {len(cols):>2} features  val MAE {s['MAE']:.2f}  ({fit_s:.0f}s)")


def summary(h):
    d = load_done(h)
    if d.empty:
        return None
    ref = d[d["config"] == "all"].set_index("model")["MAE"]
    dr = d[d["config"].str.startswith("drop_")].copy()
    dr["group"] = dr["config"].str[5:]
    dr["delta_pct"] = (dr["MAE"] / dr["model"].map(ref) - 1) * 100
    return dr.pivot(index="model", columns="group", values="delta_pct").reindex(columns=list(FEATURE_GROUPS))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for h in ev.HORIZONS:
        t = summary(h)
        if t is None:
            continue
        t = t.rename(index=mo.LABEL)
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
        from matplotlib.colors import PowerNorm
        vals = np.clip(t.to_numpy("float64"), 0, None)             # colour = size of the harm (square-root scale so
        im = ax.flat[0].imshow(vals, cmap="Reds", aspect="auto",      # small effects stay visible next to the big one)
                               norm=PowerNorm(0.5, vmin=0, vmax=max(float(np.nanmax(vals)), 1e-9)))
        ax.flat[0].set_xticks(range(t.shape[1]))
        ax.flat[0].set_xticklabels(t.columns, rotation=30, ha="right")
        ax.flat[0].set_yticks(range(t.shape[0]))
        ax.flat[0].set_yticklabels(t.index)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.iat[i, j]
                if np.isfinite(v):
                    ax.flat[0].text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8)
        ax.flat[0].set_title(f"{h}-minute horizon: MAE change (%) when one feature group is removed")
        fig.colorbar(im, ax=ax.flat[0], fraction=0.03)
        d = load_done(h)
        c = d[(d["model"] == "lgbm") & d["config"].str.startswith(("cum_", "all"))].copy()
        if not c.empty:
            c["k"] = c["config"].map(lambda s: 7 if s == "all" else int(s.split("_")[1]))
            c = c.sort_values("k")
            ax.flat[1].plot(c["k"], c["MAE"], "o-")
            ax.flat[1].set_xticks(c["k"])
            ax.flat[1].set_xticklabels([BUILD_ORDER[k - 1] for k in c["k"]], rotation=40, ha="right")
            ax.flat[1].set_ylabel("validation MAE")
            ax.flat[1].set_title("LightGBM build-up (groups added in order)")
            ax.flat[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ev.FIGS / f"ablation_h{h}.png", dpi=130)
        plt.close(fig)
        t.round(2).to_csv(OUT / f"ablation_delta_pct_h{h}.csv")
        print(f"saved {ev.FIGS / f'ablation_h{h}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--h", nargs="+", type=int, default=[15, 60])
    ap.add_argument("--rows", type=int, default=None, help="override the training-row cap (testing)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="threads for the fits (default: half the cores, so the laptop stays usable)")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        names = mo.MODELS if a.models == ["all"] else a.models
        t0 = time.time()
        run(names, a.h, a.rows, a.jobs)
        print(f"\nfits finished in {(time.time() - t0) / 60:.1f} min")
    plot()


if __name__ == "__main__":
    main()
