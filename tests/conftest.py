import json, sys, time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def root():
    return ROOT


def pytest_sessionfinish(session, exitstatus):
    """Write a small status file that the Gate G1 notebook reads."""
    out = ROOT / ("results_smoke" if __import__("os").environ.get("TP_SMOKE") == "1" else "results")
    out.mkdir(exist_ok=True)
    (out / "test_status.json").write_text(json.dumps({
        "all_passed": exitstatus == 0,
        "tests_collected": session.testscollected,
        "tests_failed": session.testsfailed,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
