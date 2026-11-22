"""LEAKAGE TEST 1  -  "scramble the future".

We replace ALL flow and weather data at and after a cut time with random numbers (keeping the pattern of missing
values), rebuild the feature table, and require that every feature of every origin at or before the cut is
IDENTICAL to the table built from the real data. If any feature peeked even one bin into the future, it would
change and this test fails.

Positive control: scrambling data in the PAST must change later features, proving the test can see changes.

Run from the repo root:   python -m pytest tests/test_leakage_scramble.py -v -s
"""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src import features as fe

CUT = pd.Timestamp("2018-03-15 12:30")          # mid-hour on purpose: a next-hour weather leak would show up
COMPARE = fe.ALL_FEATURES + ["sc"]


def _scramble_flow(F, start, end, seed):
    rng = np.random.default_rng(seed)
    m = (F.index >= start) & (F.index < end)
    block = F.loc[m].to_numpy()
    vals = rng.uniform(0, 2500, size=block.shape).astype("float32")
    vals[np.isnan(block)] = np.nan               # keep which readings are missing
    G = F.copy()
    G.loc[m] = vals
    return G


def _scramble_weather(W, start, seed):
    rng = np.random.default_rng(seed)
    W = W.copy()
    m = (W["time"] > start).to_numpy()
    W.loc[m, "precip_mm"] = rng.exponential(2.0, m.sum())
    W.loc[m, "temp_c"] = rng.uniform(-5, 40, m.sum())
    W.loc[m, "wind"] = rng.uniform(0, 60, m.sum())
    return W


def _build_in(tmp, h, flow, weather, root):
    d = tmp / "data"
    d.mkdir(parents=True, exist_ok=True)
    flow.to_parquet(d / "flow_15min_100.parquet")
    weather.to_csv(d / "weather_hourly.csv", index=False)
    shutil.copy(root / "data" / "sensors_100.csv", d / "sensors_100.csv")
    old = os.getcwd()
    os.chdir(tmp)                                # build_features reads relative paths "data/..."
    try:
        return fe.build_features(h)
    finally:
        os.chdir(old)


@pytest.fixture(scope="module")
def raw(root):
    F = pd.read_parquet(root / "data" / "flow_15min_100.parquet")
    W = pd.read_csv(root / "data" / "weather_hourly.csv", parse_dates=["time"])
    return F, W


@pytest.fixture(scope="module")
def real_tables(root):
    old = os.getcwd()
    os.chdir(root)
    try:
        return {h: fe.build_features(h) for h in (15, 60)}
    finally:
        os.chdir(old)


@pytest.mark.parametrize("h", [15, 60])
def test_future_scramble_changes_nothing(h, raw, real_tables, root, tmp_path):
    F, W = raw
    far = F.index[-1] + pd.Timedelta(days=1)
    scr = _build_in(tmp_path, h, _scramble_flow(F, CUT, far, seed=1), _scramble_weather(W, CUT, seed=2), root)
    a = real_tables[h]
    a = a[a["time"] <= CUT].reset_index(drop=True)
    b = scr[scr["time"] <= CUT].reset_index(drop=True)
    assert len(a) > 100_000, "cut is too early to be a meaningful test"
    assert len(a) == len(b), "row set changed when only the future was scrambled"
    assert (a["ID2"].to_numpy() == b["ID2"].to_numpy()).all() and (a["time"].to_numpy() == b["time"].to_numpy()).all()
    bad = {}
    for c in COMPARE:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if not np.array_equal(x, y, equal_nan=True):
            diff = ~np.isclose(x, y, rtol=0, atol=0, equal_nan=True)
            bad[c] = (int(diff.sum()), float(np.nanmax(np.abs(x - y))))
    assert not bad, f"features that changed when the FUTURE was scrambled (count, max diff): {bad}"
    last = a["time"].max()
    print(f"\n  h={h}: {len(a):,} rows with origin <= {CUT}; all {len(COMPARE)} columns identical (latest origin {last})")


def test_positive_control_past_scramble_changes_features(raw, real_tables, root, tmp_path):
    F, W = raw
    start, end = CUT - pd.Timedelta(hours=6), CUT
    scr = _build_in(tmp_path, 15, _scramble_flow(F, start, end, seed=3), W, root)
    a, b = real_tables[15], scr
    win = lambda t: t[(t["time"] >= CUT) & (t["time"] < CUT + pd.Timedelta(hours=3))].reset_index(drop=True)
    A, B = win(a), win(b)
    assert len(A) == len(B) and len(A) > 100
    changed = (A["mean_24h"].to_numpy() != B["mean_24h"].to_numpy()).mean()
    assert changed > 0.95, f"only {changed:.0%} of rows changed: the test would not notice a leak"
    print(f"\n  positive control: {changed:.0%} of {len(A)} rows changed after scrambling the past (test is sensitive)")
