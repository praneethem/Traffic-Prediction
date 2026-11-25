"""The flow tables have no missing or duplicated timestamps, so every gap in the data is a missing value, not a missing row.

Uses the 15-minute table, which is committed to Git; the 5-minute table is checked too when it is present."""
import pandas as pd
import pytest


def _check(path, step, rows_per_year):
    F = pd.read_parquet(path)
    assert F.index.is_unique, "duplicated timestamps"
    assert F.index.is_monotonic_increasing
    gaps = F.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta(minutes=step)).all(), "missing or irregular timestamps"
    for year in (2017, 2018, 2019):
        assert (F.index.year == year).sum() == rows_per_year       # 365 full days in each of the three years
    assert F.shape[1] == 100


def test_15min_table_is_complete_and_unique(root):
    path = root / "data" / "flow_15min_100.parquet"
    if not path.exists():
        pytest.skip("flow_15min_100.parquet not present")
    _check(path, 15, 365 * 96)


def test_5min_table_is_complete_and_unique(root):
    path = root / "data" / "flow_5min_100.parquet"
    if not path.exists():
        pytest.skip("flow_5min_100.parquet is generated, not stored in Git (README step 0)")
    _check(path, 5, 365 * 288)
