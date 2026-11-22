import numpy as np, pandas as pd

flow = pd.read_parquet("data/flow_5min_100.parquet")
sensors = pd.read_csv("data/sensors_100.csv")
lanes = sensors.set_index(sensors["ID2"].astype(str))["Lanes"].reindex(flow.columns).to_numpy()
X = flow.to_numpy(dtype="float64")
hour = flow.index.hour.to_numpy()
day = (hour >= 6) & (hour < 22)
year = flow.index.year.to_numpy()

def long_zero_runs(mask, min_len):
    change = np.r_[True, mask[1:] != mask[:-1]]
    gid = np.cumsum(change) - 1
    size = np.bincount(gid)[gid]
    return mask & (size >= min_len)

def identical_runs(x, min_len):
    change = np.r_[True, x[1:] != x[:-1]]
    gid = np.cumsum(change) - 1
    size = np.bincount(gid)[gid]
    return (size >= min_len) & ~np.isnan(x) & (x != 0)

zero_day = (X == 0) & day[:, None]
print("Daytime zero share of all readings:", round(float(zero_day.mean()), 5))
print("\nMin stretch | zero-run share | identical-run share")
keep = {}
for name, L in {"1h": 12, "2h": 24, "4h": 48}.items():
    fz = np.zeros(X.shape, bool); fi = np.zeros(X.shape, bool)
    for j in range(X.shape[1]):
        fz[:, j] = long_zero_runs(zero_day[:, j], L)
        fi[:, j] = identical_runs(X[:, j], L)
    print(f"{name:>11} | {fz.mean():.5f}        | {fi.mean():.5f}")
    if name == "2h":
        keep["fz"], keep["fi"] = fz, fi

fz, fi = keep["fz"], keep["fi"]
print("\nShare of daytime zeros that sit inside 2h+ runs:", round(float(fz.sum() / zero_day.sum()), 3))

high = X > 200 * lanes[None, :]
print("Above 200 per lane per 5 min:", int(high.sum()), "readings,", round(float(high.mean()), 6))

flagged = fz | fi | high
print("\nAll three rules together (2h):", round(float(flagged.mean()), 5))
for y in (2017, 2018, 2019):
    print(y, "flagged share:", round(float(flagged[year == y].mean()), 5))

per = flagged.mean(axis=0)
print("Sensors with more than 1% flagged:", int((per > 0.01).sum()), "of", X.shape[1])
top = np.argsort(per)[::-1][:5]
print("Five most affected sensors (ID2, corridor, flagged share):")
for j in top:
    row = sensors[sensors["ID2"].astype(str) == flow.columns[j]].iloc[0]
    print(" ", flow.columns[j], row["corridor"], round(float(per[j]), 4))
