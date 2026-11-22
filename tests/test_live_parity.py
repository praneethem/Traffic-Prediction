"""LEAKAGE TEST 2  -  "one-row function must equal the table".

src/live.py rebuilds the features for ONE origin time using only the history that ended before that origin
(it is physically not given any later data). We check it against the saved feature table for random origins
in train, validation and test. Equality (to float32 precision) means the table uses nothing but the past, and it also
proves the app computes the same features the models were trained on.

Run from the repo root (after python -m src.features):   python -m pytest tests/test_live_parity.py -v -s
"""
import numpy as np
import pandas as pd
import pytest

from src import features as fe
from src.live import live_features, WINDOW_BINS

BIN = pd.Timedelta(minutes=15)
COMPARE = fe.ALL_FEATURES + ["sc"]


@pytest.fixture(scope="module")
def raw(root):
    F = pd.read_parquet(root / "data" / "flow_15min_100.parquet")
    S = pd.read_csv(root / "data" / "sensors_100.csv")
    W = pd.read_csv(root / "data" / "weather_hourly.csv", parse_dates=["time"])
    return F, S, W


@pytest.mark.parametrize("h", [15, 60])
def test_live_equals_table(h, raw, root):
    F, S, W = raw
    table = pd.read_parquet(root / "data" / f"features_h{h}.parquet")
    table["time"] = pd.to_datetime(table["time"])
    rng = np.random.default_rng(11)
    origins = []
    for split in ("train", "val", "test"):
        times = table.loc[table["split"] == split, "time"].unique()
        origins += list(rng.choice(times, size=12, replace=False))
    # also the very first origins of the table (shortest history) and a holiday
    origins += [table["time"].min(), pd.Timestamp("2018-12-25 09:15"), pd.Timestamp("2019-07-04 17:00")]
    n_cmp, worst = 0, 0.0
    for o in origins:
        o = pd.Timestamp(o)
        rows = table[table["time"] == o]
        if rows.empty:
            continue
        hist = F.loc[F.index < o].iloc[-WINDOW_BINS:]
        live = live_features(hist, S, W, o, h, allow_short=True).set_index("ID2")
        for _, r in rows.iterrows():
            lv = live.loc[int(r["ID2"])]
            assert bool(lv["ok"]), f"live features not finite where the table has a row (origin {o}, sensor {r['ID2']})"
            for c in COMPARE:
                a, b = np.float32(lv[c]), np.float32(r[c])
                if not np.isclose(a, b, rtol=1e-5, atol=1e-6):
                    pytest.fail(f"h={h} origin {o} sensor {int(r['ID2'])} feature {c}: live={a} table={b}")
                worst = max(worst, abs(float(a) - float(b)))
            n_cmp += 1
    assert n_cmp >= 300, f"too few rows compared ({n_cmp})"
    print(f"\n  h={h}: {n_cmp} rows x {len(COMPARE)} columns match (max abs diff {worst:.2e})")


def test_live_refuses_future_data(raw):
    F, S, W = raw
    o = pd.Timestamp("2018-06-01 08:00")
    with pytest.raises(ValueError):
        live_features(F.loc[F.index <= o], S, W, o, 15)       # includes the bin that starts AT the origin
