import numpy as np, pandas as pd, holidays
from src.imputation import causal_fill, scale_factors

WEEK = 672
H_OFFSET = {15: 0, 60: 3}   # target bin = origin bin + offset
SPLITS = {"train": ("2017-01-29", "2018-06-30"),
          "val": ("2018-07-01", "2018-12-31"),
          "test": ("2019-01-01", "2019-12-31")}
PURGE = 4                   # last origins of train and val are dropped (1 hour)

FEATURE_GROUPS = {
    "Lags": ["flow_t", "flow_t15", "flow_t30", "flow_t45", "flow_t60", "flow_t120", "miss_3h"],
    "Rolling": ["mean_1h", "std_1h", "mean_3h", "std_3h", "mean_24h", "scale_7d"],
    "Seasonal": ["same_slot_last_week", "same_slot_4wk_mean"],
    "Calendar": ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "month", "is_holiday", "is_weekend"],
    "Weather": ["precip_mm", "temp_c", "wind_kmh", "rain_3h_mm"],
    "Neighbours": ["nb_back", "nb_ahead", "has_back", "has_ahead"],
    "Context": ["lanes"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def neighbour_index(sensors):
    key = {(c, p): j for j, (c, p) in enumerate(zip(sensors["corridor"], sensors["position"]))}
    role = sensors["role"].to_numpy()
    back = np.full(len(sensors), -1)
    ahead = np.full(len(sensors), -1)
    for j, (c, p) in enumerate(zip(sensors["corridor"], sensors["position"])):
        for step, arr in ((-1, back), (1, ahead)):
            k = key.get((c, p + step), -1)
            if k >= 0 and role[j] == "warm" and role[k] != "warm":
                k = -1
            arr[j] = k
    return back, ahead


def calendar_arrays(times):
    mod = (times.hour * 60 + times.minute).to_numpy()
    dow = times.day_of_week.to_numpy()
    ca = holidays.country_holidays("US", subdiv="CA", years=range(2016, 2021))
    hol = pd.DatetimeIndex(sorted(ca.keys()))
    return dict(
        tod_sin=np.sin(2 * np.pi * mod / 1440), tod_cos=np.cos(2 * np.pi * mod / 1440),
        dow_sin=np.sin(2 * np.pi * dow / 7), dow_cos=np.cos(2 * np.pi * dow / 7),
        month=times.month.to_numpy().astype("float64"),
        is_holiday=np.asarray(times.normalize().isin(hol)).astype("float64"),
        is_weekend=(dow >= 5).astype("float64"))


def weather_arrays(weather, times):
    hrs = times.floor("h")
    out = {}
    for c, g in weather.groupby("corridor"):
        g = g.set_index("time").sort_index()
        rain3 = g["precip_mm"].rolling(3, min_periods=3).sum()
        out[c] = dict(precip_mm=g["precip_mm"].reindex(hrs).to_numpy(),
                      temp_c=g["temp_c"].reindex(hrs).to_numpy(),
                      wind_kmh=g["wind"].reindex(hrs).to_numpy(),
                      rain_3h_mm=rain3.reindex(hrs).to_numpy())
    return out


def build_features(h):
    off = H_OFFSET[h]
    F = pd.read_parquet("data/flow_15min_100.parquet")
    sensors = pd.read_csv("data/sensors_100.csv")
    sensors = sensors.set_index(sensors["ID2"].astype(str)).loc[F.columns].reset_index(drop=True)
    weather = pd.read_csv("data/weather_hourly.csv", parse_dates=["time"])
    G, _ = causal_fill(F)
    S = scale_factors(G)
    times = F.index
    T = len(times)
    Fv, Gv, Sv = (a.to_numpy("float64") for a in (F, G, S))

    roll = {}
    for tag, w in (("1h", 4), ("3h", 12), ("24h", 96)):
        r = G.rolling(w, min_periods=w)
        roll["mean_" + tag] = r.mean().to_numpy("float64")
        if tag != "24h":
            roll["std_" + tag] = r.std().to_numpy("float64")
    miss = F.isna().astype("float64").rolling(12, min_periods=12).sum().to_numpy("float64")

    cal = calendar_arrays(times)
    wx = weather_arrays(weather, times)
    back_idx, ahead_idx = neighbour_index(sensors)

    b = np.arange(T)
    split_of = np.full(T, "", dtype=object)
    for name, (a, z) in SPLITS.items():
        m = (times >= pd.Timestamp(a)) & (times < pd.Timestamp(z) + pd.Timedelta(days=1))
        idx = np.flatnonzero(m)
        if name != "test":
            idx = idx[:-PURGE]
        split_of[idx] = name
    valid_o = (split_of != "") & (b + off <= T - 1)

    parts, dropped = [], 0
    tnp = times.to_numpy()
    for j in range(len(sensors)):
        role = sensors.at[j, "role"]
        o = np.flatnonzero(valid_o & (((b % 96) + (b // 96) + j) % 4 == 0))
        if role == "cold":
            o = o[split_of[o] == "test"]
        if len(o) == 0:
            continue
        tau = o + off
        sc = Sv[o - 1, j]
        c = sensors.at[j, "corridor"]
        own = Gv[o - 1, j] / sc

        def nb(k):
            return np.full(len(o), np.nan) if k < 0 else Gv[o - 1, k] / Sv[o - 1, k]
        bk, ak = back_idx[j], ahead_idx[j]
        nb_back = np.where(np.full(len(o), bk >= 0), nb(bk), own)
        nb_ahead = np.where(np.full(len(o), ak >= 0), nb(ak), own)

        feat = {
            "flow_t": own, "flow_t15": Gv[o - 2, j] / sc, "flow_t30": Gv[o - 3, j] / sc,
            "flow_t45": Gv[o - 4, j] / sc, "flow_t60": Gv[o - 5, j] / sc, "flow_t120": Gv[o - 9, j] / sc,
            "miss_3h": miss[o - 1, j],
            "mean_1h": roll["mean_1h"][o - 1, j] / sc, "std_1h": roll["std_1h"][o - 1, j] / sc,
            "mean_3h": roll["mean_3h"][o - 1, j] / sc, "std_3h": roll["std_3h"][o - 1, j] / sc,
            "mean_24h": roll["mean_24h"][o - 1, j] / sc, "scale_7d": sc,
            "same_slot_last_week": Gv[tau - WEEK, j] / sc,
            "same_slot_4wk_mean": (Gv[tau - WEEK, j] + Gv[tau - 2 * WEEK, j]
                                   + Gv[tau - 3 * WEEK, j] + Gv[tau - 4 * WEEK, j]) / 4 / sc,
            "tod_sin": cal["tod_sin"][tau], "tod_cos": cal["tod_cos"][tau],
            "dow_sin": cal["dow_sin"][tau], "dow_cos": cal["dow_cos"][tau],
            "month": cal["month"][tau], "is_holiday": cal["is_holiday"][tau], "is_weekend": cal["is_weekend"][tau],
            "precip_mm": wx[c]["precip_mm"][o], "temp_c": wx[c]["temp_c"][o],
            "wind_kmh": wx[c]["wind_kmh"][o], "rain_3h_mm": wx[c]["rain_3h_mm"][o],
            "nb_back": nb_back, "nb_ahead": nb_ahead,
            "has_back": np.full(len(o), float(bk >= 0)), "has_ahead": np.full(len(o), float(ak >= 0)),
            "lanes": np.full(len(o), float(sensors.at[j, "Lanes"])),
        }
        X = np.column_stack([feat[f] for f in ALL_FEATURES])
        y_raw = Fv[tau, j]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y_raw) & np.isfinite(sc) & (sc > 0)
        dropped += int((~ok).sum())
        df = pd.DataFrame(X[ok].astype("float32"), columns=ALL_FEATURES)
        df.insert(0, "split", split_of[o][ok])
        df.insert(1, "time", tnp[o][ok])
        df.insert(2, "target_time", tnp[tau][ok])
        df.insert(3, "ID2", np.int32(sensors.at[j, "ID2"]))
        df.insert(4, "corridor", c)
        df.insert(5, "role", role)
        df["sc"] = sc[ok].astype("float32")
        df["y_raw"] = y_raw[ok].astype("float32")
        df["y"] = (y_raw[ok] / sc[ok]).astype("float32")
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    print(f"  rows dropped (missing target, scale or feature): {dropped}")
    return out


if __name__ == "__main__":
    for h in (15, 60):
        print(f"\nHorizon {h} min")
        df = build_features(h)
        df.to_parquet(f"data/features_h{h}.parquet", compression="zstd")
        print("  rows by split and role:")
        print(df.groupby(["split", "role"]).size().unstack(fill_value=0).to_string())
        print("  features:", len(ALL_FEATURES), "| any NaN in features:", bool(df[ALL_FEATURES].isna().any().any()))
        v = df[df["split"] == "val"]
        for name, col in (("Persistence", "flow_t"), ("Last week", "same_slot_last_week"),
                          ("4-week average", "same_slot_4wk_mean")):
            mae = float(np.abs(v[col] * v["sc"] - v["y_raw"]).mean())
            print(f"  Validation MAE, {name}: {mae:.1f} vehicles per 15 min")
