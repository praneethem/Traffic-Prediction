import numpy as np, pandas as pd

MIN_RUN = 24             # 2 hours of 5-minute readings
MAX_PER_LANE_5MIN = 200  # equals 600 vehicles per lane per 15 minutes


def _run_sizes(a):
    """For every position, the length of the run of equal consecutive values it belongs to."""
    change = np.r_[True, a[1:] != a[:-1]]
    gid = np.cumsum(change) - 1
    return np.bincount(gid)[gid]


def flag_faults(flow, lanes):
    """flow: DataFrame (time x sensor) of 5-minute counts. Returns a boolean DataFrame, True = faulty."""
    X = flow.to_numpy(dtype="float64")
    hour = flow.index.hour.to_numpy()
    day = (hour >= 6) & (hour < 22)
    bad = np.zeros(X.shape, dtype=bool)
    for j in range(X.shape[1]):
        x = X[:, j]
        zero_day = (x == 0) & day
        bad[:, j] |= zero_day & (_run_sizes(zero_day) >= MIN_RUN)
        bad[:, j] |= (_run_sizes(x) >= MIN_RUN) & ~np.isnan(x) & (x != 0)
        bad[:, j] |= x > MAX_PER_LANE_5MIN * lanes[j]
    return pd.DataFrame(bad, index=flow.index, columns=flow.columns)


def to_15min(flow_clean):
    """5-minute counts (faults already NaN) -> vehicles per 15 minutes, labelled by bin start.
    Value = mean of valid readings x 3, only if at least 2 of the 3 readings are valid."""
    g = flow_clean.groupby(flow_clean.index.floor("15min"))
    out = (g.mean() * 3).where(g.count() >= 2)
    return out.astype("float32")


if __name__ == "__main__":
    flow = pd.read_parquet("data/flow_5min_100.parquet")
    sensors = pd.read_csv("data/sensors_100.csv")
    lanes = sensors.set_index(sensors["ID2"].astype(str))["Lanes"].reindex(flow.columns).to_numpy()

    bad = flag_faults(flow, lanes)
    clean = flow.mask(bad)
    f15 = to_15min(clean)
    f15.to_parquet("data/flow_15min_100.parquet", compression="zstd")

    print("5-minute readings flagged as faulty:", round(float(bad.to_numpy().mean()), 5))
    print("15-minute shape:", f15.shape, "|", f15.index.min(), "->", f15.index.max())
    g = clean.groupby(clean.index.floor("15min"))
    print("Every 15-minute bin has exactly 3 readings:", bool((g.size() == 3).all()))
    full = (g.count() == 3).to_numpy()
    diff = np.abs(f15.to_numpy()[full] - g.sum().to_numpy()[full]).max()
    print("Max difference from a plain sum where all 3 readings are valid:", float(diff))
    raw15 = to_15min(flow)
    for y in (2017, 2018, 2019):
        a = float(raw15.loc[str(y)].isna().to_numpy().mean())
        b = float(f15.loc[str(y)].isna().to_numpy().mean())
        print(y, "15-min NaN share: before cleaning", round(a, 5), "| after cleaning", round(b, 5))
    v = f15.to_numpy()
    print("15-min flow per lane, 99.9th percentile and max:",
          np.nanpercentile(v / lanes[None, :], 99.9).round(1), np.nanmax(v / lanes[None, :]).round(1))
