"""Analysis D: robustness and sensitivity (validation only, 2019 is never opened).

Models are fitted once per horizon on ALL train rows with their frozen best configuration, then stressed four ways:

  noise    : multiplicative Gaussian noise (5 %, 10 %, 20 %) on the five lag readings flow_t ... flow_t120
             (rolling and seasonal features are left clean, so this UNDER-states the harm of a noisy sensor)
  missing  : the latest reading is missing for 10 %, 30 %, 50 % of rows and replaced by the reading before it
             (carry-forward, what a deployed system would do): flow_t := flow_t15
  history  : refit using only the last 1, 3, 6, 12 months of the training period (18 months = all) -> reduced history
  sensors  : refit on a random half of the warm sensors, score on the OTHER half (sensors never seen in training)
             and compare with the full model on the same held-out sensors

  python scripts/run_robustness.py --h 15 60
  python scripts/run_robustness.py --h 15 60 --models ridge dt lgbm xgb
  python scripts/run_robustness.py --h 15 60 --models ada --rows 100000    # slow models: capped rows, merged into the CSV
  python scripts/run_robustness.py --plot-only
Results: results/robustness/robust_h{h}.csv and results/figures/robust_h{h}.png.  --rows N caps training rows (testing)."""
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
from src.features import ALL_FEATURES   # noqa: E402

OUT = ev.RESULTS / "robustness"
OUT.mkdir(exist_ok=True)
LAGS = ["flow_t", "flow_t15", "flow_t30", "flow_t45", "flow_t60", "flow_t120"]
NOISE = (0.05, 0.10, 0.20)
MISSING = (0.10, 0.30, 0.50)
MONTHS = (1, 3, 6, 12)
SEED = 7
HISTORY_MODELS = ("ridge", "dt", "lgbm")       # cheap enough to refit 4 more times


def fit_full(name, h, tr, jobs):
    best = ev.read_best(name, h)
    if best is None:
        return None
    return fit_on(name, best, tr, jobs)


def _dtype(m):
    """Linear pipelines (Ridge, PCA+Ridge) are fitted and scored in float64: mean_1h is an exact linear combination of
    four lag columns, so in float32 the weight on that direction is set by rounding, which does not matter on clean data
    but changes the response to noise on the lags (Ridge and PCA+Ridge then differ by about 1.3 MAE)."""
    return "float64" if hasattr(m, "steps") else "float32"


def fit_on(name, best, df, jobs):
    p = dict(best["params"])
    m = mo.make_model(name, p, n_estimators=best.get("best_iter"), early_stop=False, n_jobs=jobs)
    dt = _dtype(m)
    m, _, _ = mo.fit_model(name, m, df[ALL_FEATURES].to_numpy(dt), df["y"].to_numpy(dt))
    return m


def score(m, va):
    pv = m.predict(va[ALL_FEATURES].to_numpy(_dtype(m))).astype("float64") * va["sc"].to_numpy("float64")
    return M.score(va["y_raw"], pv)


def run(models, horizons, rows, jobs):
    for h in horizons:
        tr = ev.load_table(h, ["train"])
        n_all = len(tr)
        all_sensors = np.array(sorted(tr["ID2"].unique()))      # fixed before any row cap
        if rows:
            tr = mo.month_stratified_subset(tr, rows)
        va = ev.load_table(h, ["val"])
        rec = []
        fitted = {}
        for name in models:
            t0 = time.time()
            rng = np.random.default_rng([SEED, h, mo.MODELS.index(name)])   # own stream: results do not depend on which models run together
            m = fit_full(name, h, tr, jobs)
            if m is None:
                print(f"[{name} h={h}] skipped: no tuned config")
                continue
            fitted[name] = m
            base = score(m, va)
            rec.append(dict(model=name, h=h, experiment="baseline", setting="none", MAE=base["MAE"], WAPE=base["WAPE"]))
            print(f"[{name} h={h}] baseline val MAE {base['MAE']:.2f}  (fit {time.time() - t0:.0f}s)")
            for s in NOISE:                                   # 1. noisy readings
                v = va.copy()
                for c in LAGS:
                    v[c] = v[c] * (1 + s * rng.standard_normal(len(v)))
                r = score(m, v)
                rec.append(dict(model=name, h=h, experiment="noise", setting=f"{int(s * 100)}%", MAE=r["MAE"], WAPE=r["WAPE"]))
            for p_miss in MISSING:                            # 2. latest reading missing -> carry forward
                v = va.copy()
                hit = rng.random(len(v)) < p_miss
                v.loc[hit, "flow_t"] = v.loc[hit, "flow_t15"]
                r = score(m, v)
                rec.append(dict(model=name, h=h, experiment="missing", setting=f"{int(p_miss * 100)}%", MAE=r["MAE"], WAPE=r["WAPE"]))
            print(f"[{name} h={h}] noise and missing-reading tests done")

        t_end = tr["time"].max()
        for name in [n for n in models if n in HISTORY_MODELS and n in fitted]:   # 3. reduced training history
            best = ev.read_best(name, h)
            for mths in MONTHS:
                sub = tr[tr["time"] > t_end - pd.DateOffset(months=mths)]
                if len(sub) < 1000:
                    continue
                r = score(fit_on(name, best, sub, jobs), va)
                rec.append(dict(model=name, h=h, experiment="history", setting=f"{mths} mo", MAE=r["MAE"], WAPE=r["WAPE"]))
            full = [x for x in rec if x["model"] == name and x["experiment"] == "baseline"][0]
            rec.append(dict(model=name, h=h, experiment="history", setting="all", MAE=full["MAE"], WAPE=full["WAPE"]))
            print(f"[{name} h={h}] reduced-history tests done")

        sensors = all_sensors                                                  # 4. unseen sensors
        keep = set(np.random.default_rng([SEED, h]).choice(sensors, size=len(sensors) // 2, replace=False).tolist())   # same half for every model
        tr_half = tr[tr["ID2"].isin(keep)]
        va_out = va[~va["ID2"].isin(keep)]
        for name in fitted:
            best = ev.read_best(name, h)
            same = score(fitted[name], va_out)                                 # full model, held-out sensors
            half = score(fit_on(name, best, tr_half, jobs), va_out)            # half-sensor model, same sensors
            rec.append(dict(model=name, h=h, experiment="sensors", setting="trained on all", MAE=same["MAE"], WAPE=same["WAPE"]))
            rec.append(dict(model=name, h=h, experiment="sensors", setting="trained on half (unseen)", MAE=half["MAE"], WAPE=half["WAPE"]))
        print(f"[h={h}] sensor-holdout tests done ({len(va_out):,} validation rows from {len(sensors) - len(keep)} unseen sensors)")
        df = pd.DataFrame(rec)
        df["train_rows"] = len(tr)
        path = OUT / f"robust_h{h}.csv"
        if path.exists():                                   # keep the models that were not rerun
            old = pd.read_csv(path)
            old = old[~old["model"].isin(df["model"].unique())]
            if "train_rows" not in old.columns:
                old["train_rows"] = n_all                   # earlier runs used every training row
            df = pd.concat([old[df.columns.drop("MAE_change_pct", errors="ignore")], df], ignore_index=True)
        ref = df[df["experiment"] == "baseline"].set_index("model")["MAE"]
        df["MAE_change_pct"] = np.nan
        for ex in ("noise", "missing", "history"):
            k = df["experiment"] == ex
            df.loc[k, "MAE_change_pct"] = (df.loc[k, "MAE"] / df.loc[k, "model"].map(ref) - 1) * 100
        k = df["experiment"] == "sensors"
        allref = df[k & (df["setting"] == "trained on all")].set_index("model")["MAE"]
        df.loc[k, "MAE_change_pct"] = (df.loc[k, "MAE"] / df.loc[k, "model"].map(allref) - 1) * 100
        df.to_csv(path, index=False)
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for h in ev.HORIZONS:
        f = OUT / f"robust_h{h}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["model"] = d["model"].map(lambda n: mo.LABEL.get(n, n))
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, (ex, title, xl) in zip(axes, [("noise", "Noisy lag readings", "noise level"),
                                              ("missing", "Latest reading missing", "share of rows"),
                                              ("history", "Reduced training history", "months of history"),
                                              ("sensors", "Sensors not seen in training", "")]):
            e = d[d["experiment"] == ex]
            if e.empty:
                ax.set_visible(False)
                continue
            if ex == "sensors":
                piv = e.pivot(index="model", columns="setting", values="MAE")
                piv.plot.bar(ax=ax, rot=30)
                ax.set_ylabel("MAE on held-out sensors")
                ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
            else:
                order = list(dict.fromkeys(e["setting"]))
                if ex == "history":
                    order = ["1 mo", "3 mo", "6 mo", "12 mo", "all"]
                for m_, g in e.groupby("model"):
                    g = g.set_index("setting").reindex(order)          # keep positions aligned with the tick labels
                    ax.plot(range(len(g)), g["MAE_change_pct"], "o-", label=m_)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order)
                ax.set_xlabel(xl)
                ax.set_ylabel("MAE change vs clean input (%)" if ex != "history" else "MAE change vs full history (%)")
                ax.legend(fontsize=8)
            ax.set_title(f"{h} min: {title}")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ev.FIGS / f"robust_h{h}.png", dpi=130)
        plt.close(fig)
        print(f"saved {ev.FIGS / f'robust_h{h}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ridge", "dt", "lgbm", "xgb"])
    ap.add_argument("--h", nargs="+", type=int, default=[15, 60])
    ap.add_argument("--rows", type=int, default=None, help="cap training rows (testing)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot()
    else:
        t0 = time.time()
        run(a.models, a.h, a.rows, a.jobs)
        print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
