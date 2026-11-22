"""Check the repeated-template finding on your own files (reads 2017-2018 only; 2019 is never opened).

Run from the repo root:  python scripts/check_templates.py
Optional:                python scripts/check_templates.py path\\to\\data
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import quality as Q

D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
LAST15, LAST5 = "2018-12-31 23:45", "2018-12-31 23:55"
pd.set_option("display.width", 200)

F = pd.read_parquet(D / "flow_15min_100.parquet").loc[:LAST15]
S = pd.read_csv(D / "sensors_100.csv"); S["ID2"] = S["ID2"].astype(str); S = S.set_index("ID2").loc[F.columns]

print("\n1) Exact repeat of the same slot 7 days earlier, raw 5-minute file vs our 15-minute file (top 10)")
R = pd.read_parquet(D / "flow_5min_100.parquet").loc[:LAST5]
rep = lambda X, lag: ((X == X.shift(lag)) & X.notna()).sum() / X.notna().sum() * 100
t = pd.DataFrame({"repeat_5min_%": rep(R, 7 * 288), "repeat_15min_%": rep(F, 7 * 96), "corridor": S["corridor"], "position": S["position"]})
print(t.sort_values("repeat_5min_%", ascending=False).head(10).round(2))
print("median over all sensors:", t["repeat_5min_%"].median().round(2), "(5 min) |", t["repeat_15min_%"].median().round(2), "(15 min)")

print("\n2) Sensor pairs that are numerically identical (share of bins exactly equal)")
A = F.to_numpy(); cols = list(F.columns); rows = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        ok = ~np.isnan(A[:, i]) & ~np.isnan(A[:, j]) & (A[:, i] > 0)
        if ok.sum() > 1000:
            e = (A[ok, i] == A[ok, j]).mean() * 100
            if e > 50: rows.append((cols[i], cols[j], round(e, 2)))
print(pd.DataFrame(rows, columns=["sensor_a", "sensor_b", "identical_%"]).sort_values("identical_%", ascending=False).head(12).to_string(index=False))
print("sensors in identical groups:", sorted({r[0] for r in rows} | {r[1] for r in rows}))

print("\n3) Template days under the frozen rule")
fl = Q.template_days(F)
sh = (fl.mean() * 100).round(1); sh = sh[sh > 0].sort_values(ascending=False)
print(pd.DataFrame({"corridor": S.loc[sh.index, "corridor"], "position": S.loc[sh.index, "position"], "role": S.loc[sh.index, "role"], "flagged_days_%": sh}).head(15).to_string())
print(f"sensors with any flagged day: {len(sh)} | sensor-days flagged: {fl.to_numpy().mean():.2%} | cold sensors with >5% flagged days: {int(((fl.mean() > 0.05) & (S['role'] == 'cold')).sum())}")

print("\n4) Effect on validation baselines (feature tables, validation rows only)")
for h in (15, 60):
    cols_ = ["split", "target_time", "ID2", "y_raw", "sc", "flow_t", "same_slot_last_week", "same_slot_4wk_mean"]
    X = pd.read_parquet(D / f"features_h{h}.parquet", columns=cols_)
    X = X[X["split"].isin(["train", "val"])].copy(); X["target_time"] = pd.to_datetime(X["target_time"])
    mk = Q.row_mask(X, fl); isval = (X["split"] == "val").to_numpy()
    print(f"h={h}: rows flagged  train {100 * (1 - mk[~isval].mean()):.1f}%  val {100 * (1 - mk[isval].mean()):.1f}%")
    V, mv = X[isval], mk[isval]
    for name, c in {"Persistence": "flow_t", "Last week": "same_slot_last_week", "4-week average": "same_slot_4wk_mean"}.items():
        e = (V[c] * V["sc"] - V["y_raw"]).abs()
        print(f"   {name:15s} MAE all {e.mean():7.2f} | clean {e[mv].mean():7.2f} | flagged rows only {e[~mv].mean():7.2f}")
