"""Put our 2019 test results next to the published LargeST Greater Bay Area (GBA) benchmark.

NOT a like-for-like comparison (see the notes printed at the end), so only SCALE-FREE quantities are compared:
the error relative to the persistence baseline ("Historical Last" in LargeST) and MAPE.

Published numbers: Liu et al., "LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting", NeurIPS 2023
Datasets and Benchmarks, Table 2, GBA block, arXiv:2306.08259v2 (the first arXiv version, v1, lists slightly different
values for the neural models; Historical Last is identical) (2,352 sensors; 12 steps of 15 min in, 12 out; H3 = 45 min, H6 = 90 min).

  python scripts/compare_published.py        # reads saved test predictions, writes results/final/published_comparison.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev          # noqa: E402

# (MAE, MAPE %) at H3 (45 min) and H6 (90 min), copied from the paper's Table 2, GBA
PUBLISHED = {
    "Historical Last (persistence)": {"H3": (32.57, 22.78), "H6": (53.79, 43.01)},
    "LSTM": {"H3": (20.38, 15.47), "H6": (27.56, 23.52)},
    "DCRNN": {"H3": (18.71, 14.72), "H6": (23.06, 20.45)},
    "Graph WaveNet": {"H3": (17.85, 13.92), "H6": (21.11, 17.79)},
    "D2STGNN (best at H3)": {"H3": (17.54, 12.12), "H6": (20.92, 14.89)},
}


def ours(h):
    d = ev.load_preds("test", "lgbm", h)
    y, p = d["y_raw"].to_numpy("float64"), d["pred"].to_numpy("float64")
    pers = (d["flow_t"] * d["sc"]).to_numpy("float64")
    m = y > 0                                                   # MAPE needs a non-zero truth
    mape = lambda q: float(np.mean(np.abs(y[m] - q[m]) / y[m]) * 100)
    return dict(MAE=float(np.abs(y - p).mean()), MAE_persist=float(np.abs(y - pers).mean()),
                MAPE=mape(p), MAPE_persist=mape(pers), mean_flow=float(y.mean()))


def main():
    rows = []
    for k, step in (("H3", "45 min"), ("H6", "90 min")):
        hl_mae, hl_mape = PUBLISHED["Historical Last (persistence)"][k]
        for name, v in PUBLISHED.items():
            mae, mape = v[k]
            rows.append(dict(source="LargeST GBA (published)", horizon=step, method=name, MAE=mae, MAPE_pct=mape,
                             MAE_vs_persistence=mae / hl_mae, MAPE_vs_persistence=mape / hl_mape))
    for h in ev.HORIZONS:
        o = ours(h)
        rows.append(dict(source="this project (100 sensors, 2019 test)", horizon=f"{h} min", method="LightGBM",
                         MAE=o["MAE"], MAPE_pct=o["MAPE"], MAE_vs_persistence=o["MAE"] / o["MAE_persist"],
                         MAPE_vs_persistence=o["MAPE"] / o["MAPE_persist"]))
        rows.append(dict(source="this project (100 sensors, 2019 test)", horizon=f"{h} min", method="persistence",
                         MAE=o["MAE_persist"], MAPE_pct=o["MAPE_persist"], MAE_vs_persistence=1.0, MAPE_vs_persistence=1.0))
    out = pd.DataFrame(rows)
    out.to_csv(ev.FINAL / "published_comparison.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("\nNot like-for-like: LargeST uses all 2,352 GBA sensors with a graph over the whole network, 2019 only (6:2:2 split),"
          "\n12 steps of history and a 12-step output; we use 100 well-instrumented corridor sensors, 2017-2019, one horizon per model."
          "\nOur sensors carry far more traffic (mean flow ~916 per 15 min), so absolute MAE is not comparable;"
          "\nrelative-to-persistence error and MAPE are. Our 60 min horizon lies between their 45 and 90 min columns."
          "\nTheir MAPE handling of zero flow is not stated; ours excludes rows with a true flow of 0.")


if __name__ == "__main__":
    main()
