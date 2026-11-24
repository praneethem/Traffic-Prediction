"""Build the raw inputs of the project from the public LargeST download.

Input  (from https://www.kaggle.com/datasets/liuxu77/largest, see README "Get the data"):
    <largest_dir>/ca_meta.csv
    <largest_dir>/ca_his_raw_2017.h5, ca_his_raw_2018.h5, ca_his_raw_2019.h5

Output (written to --out, default data/):
    d4_quality.csv          per-sensor data quality of every District 4 sensor (2017 and 2018)
    sensors_100.csv         the 100 selected sensors (4 corridors x 25; first 20 warm, last 5 cold)
    flow_5min_100.parquet   raw 5-minute flow of those 100 sensors, 2017-2019

Steps
    quality   NaN share and daytime-zero share per District 4 sensor and year
    select    corridor candidates by a window search, then the four corridors used in the project
    extract   read the 100 columns from the three HDF5 files

Run all three steps:
    python scripts/prepare_data.py --largest data/largest --out data

Run only the selection when ca_meta.csv and d4_quality.csv already exist (takes seconds):
    python scripts/prepare_data.py --steps select --largest data/largest --out data
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

# ---- selection settings (identical to the Colab notebook) -------------------------------
WINDOW = 25            # sensors per corridor
GAP_KM = 3.0           # largest allowed distance between neighbouring sensors
MIN_USABLE = 0.85      # minimum usable share (non-NaN and not a daytime zero), both years
CORE_COUNTIES = ["Alameda", "Contra Costa", "San Francisco", "San Mateo", "Santa Clara"]
DAY_START, DAY_END = 6, 22      # hours counted as daytime when looking for zeros

# name: (first ID2, last ID2, expected Fwy). The first 20 sensors of each corridor are warm,
# the last 5 are cold (they appear only in the test rows).
CORRIDORS = {
    "C1_I580-E":  (2175, 2199, "I580-E"),
    "C2_US101-N": (1416, 1440, "US101-N"),
    "C3_I880-S":  (2795, 2819, "I880-S"),
    "C4_SR24-W":  (717,  741,  "SR24-W"),
}
SENSOR_COLUMNS = ["corridor", "position", "role", "ID2", "ID", "Fwy", "Direction", "County",
                  "Lanes", "Lat", "Lng", "usable_2017", "usable_2018", "usable_min"]
BLOCK = 10000          # rows per HDF5 read


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---- step 1: data quality ---------------------------------------------------------------
def build_quality(largest: str, out: str) -> pd.DataFrame:
    import h5py
    meta = pd.read_csv(os.path.join(largest, "ca_meta.csv"))
    d4 = meta[meta["District"] == 4].sort_values("ID2").reset_index(drop=True)
    cols = d4["ID2"].values          # ID2 equals the column position in the HDF5 files
    res = d4[["ID2", "ID", "Fwy", "Direction", "Lanes", "Lat", "Lng"]].copy()
    for y in (2017, 2018):
        t0 = time.time()
        nan = np.zeros(len(cols))
        dz = np.zeros(len(cols))
        n = 0
        with h5py.File(os.path.join(largest, f"ca_his_raw_{y}.h5"), "r") as f:
            hour = pd.to_datetime(f["t/axis1"][:]).hour.values
            dset = f["t/block0_values"]
            for r0 in range(0, dset.shape[0], BLOCK):
                r1 = min(r0 + BLOCK, dset.shape[0])
                x = dset[r0:r1, :][:, cols]
                day = ((hour[r0:r1] >= DAY_START) & (hour[r0:r1] < DAY_END))[:, None]
                nan += np.isnan(x).sum(0)
                dz += ((x == 0) & day).sum(0)
                n += r1 - r0
        res[f"nan_{y}"] = nan / n
        res[f"dayzero_{y}"] = dz / n
        res[f"usable_{y}"] = 1 - res[f"nan_{y}"] - res[f"dayzero_{y}"]
        print(f"quality {y}: {time.time() - t0:.0f} s")
    res["usable_min"] = res[["usable_2017", "usable_2018"]].min(axis=1)
    res.to_csv(os.path.join(out, "d4_quality.csv"), index=False)
    return res


# ---- step 2: selection ------------------------------------------------------------------
def candidate_windows(q: pd.DataFrame) -> pd.DataFrame:
    """For every freeway direction with at least 25 sensors, the best window of 25 consecutive
    sensors that lie in the five core counties, all have usable share >= 0.85 and are at most
    3 km apart. Best = highest mean usable share, then smallest largest gap. This is the table
    from which the four corridors were chosen."""
    q = q.copy()
    core = {c.lower() for c in CORE_COUNTIES}
    q["core"] = q["County"].str.lower().str.strip().isin(core)
    rows = []
    for (fwy, _), g in q.groupby(["Fwy", "Direction"]):
        g = g.sort_values("ID2").reset_index(drop=True)
        if len(g) < WINDOW:
            continue
        gap = haversine_km(g["Lat"].values[:-1], g["Lng"].values[:-1],
                           g["Lat"].values[1:], g["Lng"].values[1:])
        u, c = g["usable_min"].values, g["core"].values
        best, nvalid = None, 0
        for i in range(len(g) - WINDOW + 1):
            if c[i:i + WINDOW].all() and (u[i:i + WINDOW] >= MIN_USABLE).all() \
                    and (gap[i:i + WINDOW - 1] <= GAP_KM).all():
                nvalid += 1
                key = (u[i:i + WINDOW].mean(), -gap[i:i + WINDOW - 1].max())
                if best is None or key > best[0]:
                    best = (key, i)
        if best is None:
            continue
        i = best[1]
        w = g.iloc[i:i + WINDOW]
        rows.append(dict(
            Fwy=fwy, group_n=len(g), valid_windows=nvalid,
            first_ID2=int(w["ID2"].iloc[0]), last_ID2=int(w["ID2"].iloc[-1]),
            mean_usable=round(float(u[i:i + WINDOW].mean()), 4),
            max_gap_km=round(float(gap[i:i + WINDOW - 1].max()), 2),
            length_km=round(float(gap[i:i + WINDOW - 1].sum()), 1),
            lanes=f"{int(w['Lanes'].min())}-{int(w['Lanes'].max())}",
            county="/".join(sorted(w["County"].unique()))))
    cols = ["Fwy", "group_n", "valid_windows", "first_ID2", "last_ID2", "mean_usable", "max_gap_km",
            "length_km", "lanes", "county"]
    return pd.DataFrame(rows, columns=cols).sort_values(["mean_usable", "valid_windows"], ascending=False)


def select_sensors(largest: str, out: str) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(largest, "ca_meta.csv"))
    q = pd.read_csv(os.path.join(out, "d4_quality.csv")).merge(meta[["ID2", "County"]], on="ID2", how="left")
    parts = []
    for name, (a, b, fwy) in CORRIDORS.items():
        w = q[(q["ID2"] >= a) & (q["ID2"] <= b)].sort_values("ID2").copy()
        assert len(w) == WINDOW, (name, len(w))
        assert (w["Fwy"] == fwy).all(), name
        assert w["County"].isin(CORE_COUNTIES).all(), name
        w["corridor"] = name
        w["position"] = range(1, WINDOW + 1)
        w["role"] = ["warm"] * 20 + ["cold"] * 5
        parts.append(w)
    s = pd.concat(parts, ignore_index=True)
    assert s["ID2"].is_unique and len(s) == 100
    s = s[SENSOR_COLUMNS]
    s.to_csv(os.path.join(out, "sensors_100.csv"), index=False)
    print("sensors:", len(s), "| warm:", int((s.role == "warm").sum()), "| cold:", int((s.role == "cold").sum()),
          "| lowest usable share:", round(float(s["usable_min"].min()), 4))
    return s


# ---- step 3: extraction -----------------------------------------------------------------
def extract_flow(largest: str, out: str) -> None:
    import h5py
    s = pd.read_csv(os.path.join(out, "sensors_100.csv"))
    cols = np.sort(s["ID2"].values)
    assert len(cols) == 100
    frames = []
    for y in (2017, 2018, 2019):
        t0 = time.time()
        with h5py.File(os.path.join(largest, f"ca_his_raw_{y}.h5"), "r") as f:
            ts = pd.to_datetime(f["t/axis1"][:])
            dset = f["t/block0_values"]
            parts = [dset[r0:min(r0 + BLOCK, dset.shape[0]), :][:, cols].astype("float32")
                     for r0 in range(0, dset.shape[0], BLOCK)]
        arr = np.vstack(parts)
        frames.append(pd.DataFrame(arr, index=ts, columns=[str(c) for c in cols]))
        print(f"extract {y}: {arr.shape} in {time.time() - t0:.0f} s")
    flow = pd.concat(frames)
    flow.index.name = "time"
    path = os.path.join(out, "flow_5min_100.parquet")
    flow.to_parquet(path, compression="zstd")
    step = flow.index.to_series().diff().dropna().value_counts().to_dict()
    print("shape:", flow.shape, "| step sizes:", step)
    assert flow.equals(pd.read_parquet(path)), "parquet does not reload identically"
    print("written:", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--largest", default="data/largest", help="folder with ca_meta.csv and ca_his_raw_*.h5")
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--steps", nargs="+", default=["quality", "select", "extract"],
                    choices=["quality", "select", "extract"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if "quality" in a.steps:
        build_quality(a.largest, a.out)
    if "select" in a.steps:
        select_sensors(a.largest, a.out)
    if "extract" in a.steps:
        extract_flow(a.largest, a.out)


if __name__ == "__main__":
    main()
