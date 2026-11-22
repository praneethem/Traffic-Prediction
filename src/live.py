"""One-origin ("live") feature function.

Given ONLY the 15-minute flow history that ended before the origin time, it rebuilds the same 31 features as
src/features.py for every sensor. It is used twice:
  * tests/test_live_parity.py : the result must equal the feature table row for row. Because this function is
    physically handed no data from the future, parity proves the table does not use the future either.
  * the Streamlit app : the user picks an origin time and the app calls this function.

History needed: the 28-day median scale plus the one-week gap-fill lookback means 35 days is exactly enough;
we keep 40 days for margin."""
import numpy as np
import pandas as pd

from src.imputation import causal_fill, scale_factors
from src.features import (ALL_FEATURES, H_OFFSET, WEEK, neighbour_index, calendar_arrays, weather_arrays)

BIN = pd.Timedelta(minutes=15)
WINDOW_BINS = 40 * 96
MIN_BINS = 35 * 96


def align_sensors(sensors, columns):
    """Sensor table in the same order as the flow columns (identical to build_features)."""
    s = sensors.set_index(sensors["ID2"].astype(str)).loc[list(columns)]
    return s.reset_index(drop=True)


def live_features(F_hist, sensors, weather, origin, h, allow_short=False):
    """F_hist  : 15-min flow, index = bin START times, last bin starts at origin - 15 min, columns = ID2 strings.
       sensors : sensors_100.csv as a DataFrame.   weather : weather_hourly.csv (time, corridor, precip_mm, temp_c, wind).
       Returns one row per sensor: ID2, corridor, role, the 31 features, sc, and ok (all finite and sc > 0)."""
    origin = pd.Timestamp(origin)
    off = H_OFFSET[h]
    if F_hist.index[-1] != origin - BIN:
        raise ValueError("the last history bin must start at origin - 15 min (data from the future is not allowed)")
    F_hist = F_hist.iloc[-WINDOW_BINS:]
    if not (F_hist.index.to_series().diff().dropna() == BIN).all():
        raise ValueError("history has gaps in the 15-minute grid")
    n = len(F_hist)
    if n < MIN_BINS and not allow_short:
        raise ValueError(f"need at least {MIN_BINS} bins (35 days) of history, got {n}")

    sens = align_sensors(sensors, F_hist.columns)
    G, _ = causal_fill(F_hist)
    S = scale_factors(G)
    Fv, Gv, Sv = (a.to_numpy("float64") for a in (F_hist, G, S))
    tau = n + off                                  # position of the target bin in the (virtual) full series

    sc = Sv[n - 1]
    own = Gv[n - 1] / sc

    def roll(w):
        win = Gv[n - w:n]
        return win.mean(axis=0), win.std(axis=0, ddof=1)
    m1, s1 = roll(4)
    m3, s3 = roll(12)
    m24, _ = roll(96)

    cal = calendar_arrays(pd.DatetimeIndex([origin + off * BIN]))
    wx = weather_arrays(weather[weather["time"] <= origin.floor("h")], pd.DatetimeIndex([origin]))
    corr = sens["corridor"].to_numpy()

    def wxcol(k):
        return np.array([wx[c][k][0] for c in corr], dtype="float64")

    back_idx, ahead_idx = neighbour_index(sens)
    ratio = Gv[n - 1] / Sv[n - 1]
    nb_back = np.where(back_idx >= 0, ratio[back_idx], own)
    nb_ahead = np.where(ahead_idx >= 0, ratio[ahead_idx], own)

    feat = {
        "flow_t": own, "flow_t15": Gv[n - 2] / sc, "flow_t30": Gv[n - 3] / sc,
        "flow_t45": Gv[n - 4] / sc, "flow_t60": Gv[n - 5] / sc, "flow_t120": Gv[n - 9] / sc,
        "miss_3h": np.isnan(Fv[n - 12:n]).sum(axis=0).astype("float64"),
        "mean_1h": m1 / sc, "std_1h": s1 / sc, "mean_3h": m3 / sc, "std_3h": s3 / sc,
        "mean_24h": m24 / sc, "scale_7d": sc,
        "same_slot_last_week": Gv[tau - WEEK] / sc,
        "same_slot_4wk_mean": (Gv[tau - WEEK] + Gv[tau - 2 * WEEK] + Gv[tau - 3 * WEEK] + Gv[tau - 4 * WEEK]) / 4 / sc,
        "tod_sin": np.full(len(sens), cal["tod_sin"][0]), "tod_cos": np.full(len(sens), cal["tod_cos"][0]),
        "dow_sin": np.full(len(sens), cal["dow_sin"][0]), "dow_cos": np.full(len(sens), cal["dow_cos"][0]),
        "month": np.full(len(sens), cal["month"][0]), "is_holiday": np.full(len(sens), cal["is_holiday"][0]),
        "is_weekend": np.full(len(sens), cal["is_weekend"][0]),
        "precip_mm": wxcol("precip_mm"), "temp_c": wxcol("temp_c"),
        "wind_kmh": wxcol("wind_kmh"), "rain_3h_mm": wxcol("rain_3h_mm"),
        "nb_back": nb_back, "nb_ahead": nb_ahead,
        "has_back": (back_idx >= 0).astype("float64"), "has_ahead": (ahead_idx >= 0).astype("float64"),
        "lanes": sens["Lanes"].to_numpy("float64"),
    }
    X = np.column_stack([feat[f] for f in ALL_FEATURES])
    out = pd.DataFrame(X, columns=ALL_FEATURES)
    out.insert(0, "ID2", sens["ID2"].to_numpy().astype("int32"))
    out.insert(1, "corridor", corr)
    out.insert(2, "role", sens["role"].to_numpy())
    out["sc"] = sc
    out["ok"] = np.isfinite(X).all(axis=1) & np.isfinite(sc) & (sc > 0)
    return out
