"""Small files the app needs. Run once after notebook 04 (validation predictions must exist):
    python scripts/build_app_assets.py
Writes results/app/error_quantiles.json: the 10th and 90th percentile of the scaled validation error
(actual - predicted) / sensor scale, per horizon, for the LightGBM model trained on the training years only.
The app turns them into an 80 % interval: forecast + scale x q10 ... forecast + scale x q90.
Only the validation year is used, never 2019."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev          # noqa: E402

out = {}
for h in ev.HORIZONS:
    p = pd.read_parquet(ev.pred_path("val", "lgbm", h))
    t = ev.load_table(h, columns=["y_raw", "sc", "split"]).loc[p["row"].to_numpy()]
    assert (t["split"] == "val").all(), "only validation rows may be used here"
    e = (t["y_raw"].to_numpy("float64") - p["pred"].to_numpy("float64")) / t["sc"].to_numpy("float64")
    out[str(h)] = {"q10": float(np.quantile(e, 0.10)), "q90": float(np.quantile(e, 0.90)), "n": int(len(e))}
d = ev.RESULTS / "app"
d.mkdir(exist_ok=True)
(d / "error_quantiles.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
