"""Command-line runner for the long jobs (so they survive closing the notebook).

  python scripts/run_model.py tune  --models rf ada gb --h 15 60        # tune on validation, save best config + val predictions
  python scripts/run_model.py refit --models all --h 15 60 --seeds      # refit on train+val, predict 2019 ONCE, log cost
  add --redo to repeat a tuning run.   TP_SMOKE=1 gives a tiny dry run (PowerShell:  $env:TP_SMOKE="1")
Safe to stop and restart: finished (model, horizon) pairs are skipped."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import models as mo          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "refit"])
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--h", nargs="+", type=int, default=[15, 60])
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--seeds", action="store_true", help="refit mode: also run LightGBM with 2 extra seeds")
    a = ap.parse_args()
    names = mo.MODELS if a.models == ["all"] else a.models
    t0 = time.time()
    for name in names:
        for h in a.h:
            if a.mode == "tune":
                mo.tune_model(name, h, redo=a.redo)
            else:
                mo.refit_test(name, h)
                if a.seeds and name == "lgbm":
                    for s in (43, 44):
                        mo.refit_test(name, h, seed=s, tag=f"_seed{s}")
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
