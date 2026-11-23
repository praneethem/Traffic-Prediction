> **Historical document.** It was written before the analyses and the data-preparation scripts existed. Do not run step 0 (the kit is already in the repository). For the current run order see the README.

# Phase 2 + 3 runbook (run from C:\dev\traffic-prediction, venv active)

## 0. Install the kit (once)
    Expand-Archive -Force phase23_kit.zip -DestinationPath .
    pip install pytest
    Add-Content .gitignore "`nmodels/`nresults/preds/`nresults_smoke/`nmodels_smoke/"
This adds src/metrics.py, evaluate.py, live.py, models.py, tests/, notebooks/, scripts/. It does NOT touch your
features.py, cleaning.py or imputation.py.

## 1. Dry run of everything (10 minutes, optional but recommended)
    $env:TP_SMOKE="1"
    python -m pytest tests -q
    python scripts/run_model.py tune --models all --h 15 60
    python scripts/run_model.py refit --models all --h 15 60 --seeds
    Remove-Item Env:TP_SMOKE
Smoke mode uses tiny data/trees and writes ONLY to results_smoke/ and models_smoke/, so it can never block the real run.
Delete those two folders afterwards.

## 2. Real run, in this order
| Step | Where | Command / notebook | What you should see | Time (guess) |
|---|---|---|---|---|
| 14 | VS Code | notebooks/01_eda.ipynb, Run All | 11 figures in results/figures, results/eda_facts.csv | 5-10 min |
| 15+16 | Terminal | python -m pytest tests -v -s | 6 passed (scramble test h=15 and h=60, positive control, parity h=15 and h=60, future-refusal) | 5-10 min |
| 17+18 | VS Code | notebooks/02_baselines_gate1.ipynb | baseline table, quick LightGBM beats best baseline, GATE G1: PASSED | 5-10 min |
| 19-21, 24 | VS Code | notebooks/03_fast_models.ipynb | Ridge, PCA+Ridge, Tree, LightGBM, XGBoost, CatBoost tuned, both horizons | 1-2 h |
| 22-23 | Terminal | .\scripts\run_slow.ps1  (RF, AdaBoost, GB; can run overnight or on Colab) | one BEST line per model and horizon | hours |
| 25a | VS Code | notebooks/04_validation_analysis.ipynb | validation leaderboard, residuals, feature drivers, results/frozen_config.json | 5 min |
| 25b | Terminal | python scripts/run_model.py refit --models all --h 15 60 --seeds | 2019 predictions + cost logs (refuses to overwrite) | hours |
| 26 | VS Code | notebooks/05_gate2_test.ipynb | headline table with 95% intervals, warm vs cold, quarters, cost table, results/FROZEN.json | 5 min |

## 3. Rules the code enforces
* EDA reads 2017-2018 only. Tuning and early stopping use validation only. 2019 is read in step 25b/26 only.
* Refit predictions for 2019 are written once; re-running skips them. Delete the file by hand only if you accept
  that this breaks the "scored once" claim.
* "Best baseline" is chosen on validation per horizon; skill on 2019 is measured against that same baseline.
* If you change anything in src/features.py, rebuild the tables (python -m src.features), rerun the tests, and redo
  tuning (add --redo).

## 4. If something fails
Paste the last 15 lines of the error. Common ones: a library version difference in early stopping (src/models.py,
function fit_model) or an out-of-memory error in Random Forest (lower N_TREES["rf"] or run on Colab).
