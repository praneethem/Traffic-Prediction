"""Unit tests for the helper logic added after the main pipeline: the paired bootstrap, the sensor-window search,
and the published-number table. They use small synthetic inputs, so they run in seconds without any data files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paired_bootstrap as pb          # noqa: E402
import prepare_data as pdz             # noqa: E402


def _days(n_days=60, per_day=10):
    return np.repeat(np.arange(n_days), per_day)


def test_paired_identical_models_have_zero_difference():
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 900, 600)
    p = y + rng.normal(0, 20, 600)
    d, (lo, hi), win = pb.paired(_days(), y, p, p.copy())
    assert d == 0 and lo == 0 and hi == 0 and win == 0


def test_paired_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    y = rng.uniform(100, 900, 600)
    good = y + rng.normal(0, 10, 600)
    bad = y + rng.normal(0, 40, 600)
    d, (lo, hi), win = pb.paired(_days(), y, good, bad)      # reference = good
    assert d > 0 and lo > 0 and win == 1.0
    d2, (lo2, hi2), win2 = pb.paired(_days(), y, bad, good)  # reference = bad: sign flips
    assert d2 < 0 and hi2 < 0 and win2 == 0.0


def test_paired_difference_matches_plain_mae_difference():
    rng = np.random.default_rng(2)
    y = rng.uniform(100, 900, 500)
    a, b = y + rng.normal(0, 15, 500), y + rng.normal(0, 25, 500)
    d, _, _ = pb.paired(_days(50), y, a, b)
    assert np.isclose(d, np.abs(y - b).mean() - np.abs(y - a).mean())


def _sensor_table(n=30, bad=None):
    """One freeway direction with n sensors, 1 km apart, all in a core county."""
    lat = 37.5 + np.arange(n) * 0.009                    # about 1 km per step
    q = pd.DataFrame(dict(ID2=np.arange(n), Fwy="I1-N", Direction="N", Lanes=3, Lat=lat, Lng=-122.0,
                          County="Alameda", usable_min=0.99))
    if bad is not None:
        q.loc[bad, "usable_min"] = 0.5
    return q


def test_window_search_finds_a_valid_window_and_skips_bad_sensors():
    r = pdz.candidate_windows(_sensor_table(30))
    assert len(r) == 1 and r.loc[r.index[0], "valid_windows"] == 6          # 30 - 25 + 1 windows
    r2 = pdz.candidate_windows(_sensor_table(30, bad=[12]))                  # one bad sensor in the middle of all windows
    assert r2.empty
    r3 = pdz.candidate_windows(_sensor_table(30, bad=[2]))                   # bad sensor only excludes windows containing it
    assert r3.loc[r3.index[0], "first_ID2"] >= 3


def test_window_search_respects_county_and_gap_limits():
    q = _sensor_table(30)
    q.loc[10, "County"] = "Sacramento"
    assert pdz.candidate_windows(q).empty
    q = _sensor_table(30)
    q.loc[15:, "Lat"] += 0.1                              # an 11 km jump in the middle
    assert pdz.candidate_windows(q).empty
