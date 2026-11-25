"""Interaction tests for the Streamlit app (Streamlit's own AppTest, no browser needed).

They use only files that are committed to Git (the 100-sensor 15-minute table, the sensor and weather files and the two
LightGBM models), so they also pass on a fresh clone."""
import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
NEEDED = [ROOT / "data" / "flow_15min_100.parquet", ROOT / "models" / "lgbm_h15_final.joblib"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="app files not present")


def _run():
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    return at.run()


def _flow(metric):
    return int(metric.value.replace(",", "").split()[0])


def test_app_starts_and_forecasts():
    at = _run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Predicted flow" in labels and "Actual flow" in labels
    pred = _flow(at.metric[labels.index("Predicted flow")])
    assert 0 < pred < 5000                               # a plausible flow for a freeway sensor, in vehicles per 15 minutes


def test_default_forecast_is_stable():
    """Sensor 2175, 2019-03-12 17:00, 15 minutes ahead: the final model predicts about 1,760 (tolerance for float noise)."""
    at = _run()
    labels = [m.label for m in at.metric]
    assert abs(_flow(at.metric[labels.index("Predicted flow")]) - 1760) <= 3


def test_switch_horizon_and_stress_controls():
    at = _run()
    at.radio[0].set_value(60).run()
    assert not at.exception
    at.checkbox[0].check().run()                         # latest reading missing
    assert not at.exception
    at.slider[0].set_value(20).run()                     # 20 % noise on the recent readings
    assert not at.exception
    assert any(m.label == "Predicted flow" for m in at.metric)


def test_every_corridor_works():
    at = _run()
    for corridor in list(at.selectbox[0].options):
        at.selectbox[0].set_value(corridor).run()
        assert not at.exception, corridor


def test_missing_file_gives_a_message_not_a_traceback(monkeypatch):
    real_exists = pathlib.Path.exists
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False if self.name == "weather_hourly.csv" else real_exists(self))
    at = _run()
    assert not at.exception
    assert any("weather_hourly.csv" in e.value for e in at.error)
