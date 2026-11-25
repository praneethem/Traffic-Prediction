"""PCA study (validation only): does dimensionality reduction help any of the models?

The benchmark uses PCA only in front of Ridge, and the tuned PCA keeps all 31 components (so PCA + Ridge equals
Ridge). This script asks the question more broadly. The 31 standardised features are rotated onto their principal
components (fitted on training rows only). For each number of components k, Ridge, a Decision Tree, a Random Forest
and LightGBM are refitted on the first k components with their frozen settings, and scored on the validation year.
"No PCA" (the 31 original features) is the reference for each model.

Nothing here touches 2019: only training and validation rows are read. The frozen models, results/best and
FROZEN.json are not changed.

  python scripts/run_pca_study.py                      # both horizons, writes results/pca_study/
  python scripts/run_pca_study.py --h 15 --rows 20000 --ks 5 31 --out results_smoke/pca_study    # quick test
Safe to stop and restart: finished (horizon, model, k) rows are skipped."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.decomposition import PCA                      # noqa: E402
from sklearn.linear_model import Ridge                     # noqa: E402
from sklearn.preprocessing import StandardScaler           # noqa: E402

from src import evaluate as ev                             # noqa: E402
from src import models as mo                               # noqa: E402
from src.features import ALL_FEATURES                      # noqa: E402

KS = [2, 4, 6, 8, 10, 15, 20, 25, 31]
RF_KS = [10, 20]                       # the Random Forest and LightGBM are slow, so they run for fewer values of k
LGBM_KS = [4, 8, 15, 31]
SLOW_KS = {"rf": RF_KS, "lgbm": LGBM_KS}
LABEL = {"ridge": "Ridge", "dt": "Decision Tree", "rf": "Random Forest", "lgbm": "LightGBM"}


def load_train_val(h):
    cols = ALL_FEATURES + ["split", "time", "sc", "y", "y_raw"]
    df = pd.read_parquet(ev.DATA / f"features_h{h}.parquet", columns=cols,
                         filters=[("split", "in", ["train", "val"])])          # 2019 rows are never read
    df["time"] = pd.to_datetime(df["time"])
    return df[df["split"] == "train"], df[df["split"] == "val"]


def fit_score(name, params, Xa, ya, Xv, yv, scv):
    """Fit one model with frozen settings; returns (validation MAE in vehicles, seconds, trees or None)."""
    t0 = time.time()
    trees = None
    if name == "ridge":
        m = Ridge(alpha=params["alpha"]).fit(Xa.astype("float64"), ya.astype("float64"))
        p = m.predict(Xv.astype("float64"))
    elif name == "lgbm":
        m = mo.make_model("lgbm", params, early_stop=True)
        m, trees, _ = mo.fit_model("lgbm", m, Xa.astype("float32"), ya, Xv.astype("float32"), yv, scv, early_stop=True)
        p = m.predict(Xv.astype("float32"))
    else:
        m = mo.make_model(name, params)
        m.fit(Xa.astype("float32"), ya)
        p = m.predict(Xv.astype("float32"))
    return mo.val_mae_raw(yv, p, scv), time.time() - t0, trees


def plot(out, fig_dir):
    """Left: cumulative explained variance. Right: validation MAE against k (dashed = no PCA), per horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = pd.read_csv(Path(out) / "pca_study.csv")
    res["k"] = res["k"].astype(str)
    colour = {"ridge": "#0072B2", "dt": "#D55E00", "rf": "#009E73", "lgbm": "#CC79A7"}
    hs = sorted(res["h"].unique())
    fig, axes = plt.subplots(len(hs), 2, figsize=(11, 3.9 * len(hs)), squeeze=False)
    for r, h in enumerate(hs):
        e = pd.read_csv(Path(out) / f"explained_variance_h{h}.csv")
        ax = axes[r][0]
        ax.plot(e["component"], 100 * e["cumulative"], marker="o", ms=3, color="#444")
        for q in (90, 99):
            ax.axhline(q, color="#bbb", lw=0.8, ls=":")
        ax.set_xlabel("number of components k")
        ax.set_ylabel("cumulative variance explained (%)")
        ax.set_title(f"{h} min: variance kept by the first k components")
        ax = axes[r][1]
        for name in ("ridge", "dt", "rf", "lgbm"):
            d = res[(res["h"] == h) & (res["model"] == name)]
            if d.empty:
                continue
            raw = d.loc[d["k"] == "raw", "val_MAE"]
            p = d[d["k"] != "raw"].assign(kk=lambda x: x["k"].astype(int)).sort_values("kk")
            ax.plot(p["kk"], p["val_MAE"], marker="o", ms=4, color=colour[name], label=LABEL[name])
            if len(raw):
                ax.axhline(float(raw.iloc[0]), color=colour[name], ls="--", lw=1)
        ax.set_xlabel("number of components k (dashed line = no PCA, 31 original features)")
        ax.set_ylabel("validation MAE (vehicles per 15 min)")
        ax.set_title(f"{h} min: error against k")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "pca_study.png", dpi=150)
    print("figure written:", Path(fig_dir) / "pca_study.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", nargs="+", type=int, default=[15, 60])
    ap.add_argument("--rows", type=int, default=100_000, help="training rows (month-stratified subset; every k and the no-PCA reference use the same rows)")
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--models", nargs="+", default=["ridge", "dt", "lgbm", "rf"])
    ap.add_argument("--out", default=str(ev.RESULTS / "pca_study"))
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved CSVs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        plot(out, ev.FIGS)
        return
    csv = out / "pca_study.csv"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["h", "model", "k", "val_MAE", "fit_s", "trees"])
    seen = set(zip(done["h"], done["model"], done["k"].astype(str)))

    for h in a.h:
        tr, va = load_train_val(h)
        sub = mo.month_stratified_subset(tr, a.rows)
        Xs, ys = sub[ALL_FEATURES].to_numpy("float64"), sub["y"].to_numpy("float32")
        Xv, yv, scv = va[ALL_FEATURES].to_numpy("float64"), va["y"].to_numpy("float32"), va["sc"].to_numpy("float64")

        scaler = StandardScaler().fit(Xs)
        Zs, Zv = scaler.transform(Xs), scaler.transform(Xv)
        pca = PCA(n_components=Zs.shape[1], svd_solver="full", random_state=mo.SEED).fit(Zs)
        Ts, Tv = pca.transform(Zs), pca.transform(Zv)
        ev_df = pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                              "explained_variance_ratio": pca.explained_variance_ratio_,
                              "cumulative": np.cumsum(pca.explained_variance_ratio_)})
        ev_df.to_csv(out / f"explained_variance_h{h}.csv", index=False)
        print(f"[h={h}] train rows {len(sub)}, validation rows {len(va)}; components for 90/95/99% variance:",
              [int(np.searchsorted(ev_df['cumulative'], q) + 1) for q in (0.90, 0.95, 0.99)], flush=True)

        for name in a.models:
            params = ev.read_best(name, h)["params"]
            for k in ["raw"] + [k for k in a.ks if k in SLOW_KS.get(name, a.ks)]:
                if (h, name, str(k)) in seen:
                    continue
                if k == "raw":
                    Xa, Xb = (Zs, Zv) if name == "ridge" else (Xs, Xv)      # Ridge is always standardised, as in the benchmark
                else:
                    Xa, Xb = Ts[:, :k], Tv[:, :k]
                mae, secs, trees = fit_score(name, params, Xa, ys, Xb, yv, scv)
                row = {"h": h, "model": name, "k": k, "val_MAE": mae, "fit_s": secs, "trees": trees}
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(csv, index=False)
                seen.add((h, name, str(k)))
                print(f"[h={h}] {LABEL[name]:14s} k={str(k):>3s}  val MAE {mae:7.2f}  ({secs:5.0f}s)", flush=True)
    plot(out, ev.FIGS)
    print("DONE")


if __name__ == "__main__":
    main()
