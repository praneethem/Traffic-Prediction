# Traffic Flow Forecasting on Bay Area Freeways

Topic 4 (Traffic Prediction), machine learning course project.

We forecast freeway traffic flow **15 and 60 minutes ahead** for **100 sensors** in the San Francisco Bay Area, using the LargeST dataset (2017 to 2019). Nine models are tuned on one year of validation data, frozen, and scored once on 2019. A Streamlit app lets a user pick a sensor and time, see the forecast with an uncertainty band, and read why the model made it.

## Headline results (2019 test year, all 100 sensors)

The target is flow in vehicles per 15 minutes. MAE is in those units. Skill is the MAE reduction against the best baseline chosen on validation (Persistence at 15 minutes, 4-week average at 60 minutes).

| Model | 15 min MAE | 15 min skill | 60 min MAE | 60 min skill |
|---|---|---|---|---|
| Best baseline | 66.38 | 0 | 78.43 | 0 |
| Ridge (PCA + Ridge selected all 31 components, so identical) | 55.98 | 0.157 | 75.26 | 0.040 |
| Decision Tree | 52.50 | 0.209 | 67.12 | 0.144 |
| AdaBoost | 51.12 | 0.230 | 65.81 | 0.161 |
| Gradient Boosting | 50.78 | 0.235 | 64.31 | 0.180 |
| Random Forest | 49.55 | 0.253 | 62.35 | 0.205 |
| CatBoost | 48.96 | 0.262 | 61.58 | 0.215 |
| XGBoost | 48.73 | 0.266 | 61.29 | 0.219 |
| **LightGBM** | **48.63** | **0.267** | **61.15** | **0.220** |

Confidence intervals (day-block bootstrap), warm vs cold sensors, and quarter-by-quarter results are in `notebooks/05_gate2_test.ipynb` and `results/final/`. The same table is also reported on "clean rows" (see Limitations). Paired day-block bootstrap differences between models (`scripts/paired_bootstrap.py`, `results/final/paired_test_h*.csv`) show which gaps are larger than the day-to-day noise, and `scripts/compare_published.py` compares the scale-free numbers with the published LargeST benchmark (`results/final/published_comparison.csv`; we do not claim to beat graph neural networks, the setups differ).

## Method in brief

* **Target:** flow divided by the sensor's trailing 7-day mean, so busy and quiet sensors are comparable. Predictions are converted back to vehicles per 15 minutes before scoring.
* **Split (chronological):** train 2017-01 to 2018-06, validation 2018-07 to 2018-12, test 2019. 80 sensors appear in training ("warm"); 20 appear only in test ("cold"), which measures generalisation to unseen locations.
* **Sampling:** each sensor contributes one forecast origin in four (a fixed pattern), which gives about 1.34 million training rows and 0.87 million test rows per horizon.
* **31 features in 7 groups:** Lags, Rolling, Seasonal, Calendar, Weather, Neighbours (upstream and downstream sensor), Context. All are built from data that ended before the forecast origin.
* **Models:** Ridge, PCA + Ridge, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost. Tuned on validation only.
* **Final fit:** each model is refit on train + validation with the frozen settings and predicts 2019 once. The 22 prediction files are fingerprinted in `results/FROZEN.json` (Gate G2).
* **Leakage guards (in `tests/`):** a scramble test (shuffling the future must not change predictions), a positive control, and a parity test proving the app's live feature builder reproduces the training table row for row.

## Analyses

| Analysis | What it asks | Where |
|---|---|---|
| A. Feature ablation | Which feature groups matter? | `scripts/run_ablation.py`, `results/ablation/` |
| D. Robustness | Noise, missing readings, short history, unseen sensors (all nine models) | `scripts/run_robustness.py`, `results/robustness/` |
| E. Error analysis | Where do models fail, and do they fail together? | `notebooks/06_analysis_E_errors.ipynb`, `results/error_*` |
| Tuning sensitivity | How much could a wider search gain? (validation only, frozen models unchanged) | `scripts/run_tuning_sensitivity.py`, `results/tuning_sensitivity/` |
| PCA study | Does dimensionality reduction help any model? PCA with k components feeding Ridge, Decision Tree, Random Forest and LightGBM (validation only) | `scripts/run_pca_study.py`, `results/pca_study/`, `results/figures/pca_study.png` |

Short findings: the recent lags and the calendar carry the 15-minute forecast and the weekly pattern carries the 60-minute one; weather and neighbour readings add nothing measurable at these horizons; noisy inputs hurt the 15-minute model far more than missing readings; boosting needs about a year of history; sudden changes, holidays and rain are the hard cases; and the worst errors are mostly sensor dropouts that no feature can predict. PCA never helped: Ridge loses nothing only when all 31 components are kept (it is a rotation of the same information), and the tree models get clearly worse on rotated features (for example LightGBM 54.7 against 50.3 at 15 minutes on the study subset), so the 31 features are kept as they are.

## Repository layout

```
app/                 Streamlit app (streamlit_app.py)
src/                 features, cleaning, imputation, models, metrics, evaluation, live feature builder
scripts/             prepare_data.py, fetch_weather.py (build the data), run_model.py (tune / refit),
                     run_slow.ps1, ablation, robustness, paired_bootstrap, compare_published,
                     run_tuning_sensitivity, app assets
notebooks/           01 EDA, 02 baselines + Gate G1, 03 fast models, 04 validation analysis,
                     05 test + Gate G2, 06 error analysis
tests/               leakage, live-parity, data-integrity, unit and Streamlit app tests (20 tests)
report/              LaTeX report: main.tex, make_tables.py (builds tables and numbers from results/), figures, main.pdf
results/             tables, figures, tuned settings, FROZEN.json (the 2019 predictions and all models except the two the app uses are not in Git)
```

## Setup (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Most data files are **not stored in Git** (`/data/` is ignored; only the small files the app needs were added by force). Three files are needed in `data/`:

| File | Content |
|---|---|
| `flow_5min_100.parquet` | raw 5-minute flow for the 100 chosen sensors, 2017 to 2019 |
| `sensors_100.csv` | sensor ID (`ID2`), corridor, position, county, lanes, warm/cold role, usable share |
| `weather_hourly.csv` | hourly precipitation, temperature and wind per corridor |

`flow_15min_100.parquet`, `features_h15.parquet` and `features_h60.parquet` are generated by the steps below.

### Get the data

1. Download from the [LargeST dataset on Kaggle](https://www.kaggle.com/datasets/liuxu77/largest) (needs a free Kaggle account and API token) these four files into `data/largest/`: `ca_meta.csv`, `ca_his_raw_2017.h5`, `ca_his_raw_2018.h5`, `ca_his_raw_2019.h5`.

   ```powershell
   pip install kaggle
   foreach ($f in "ca_meta.csv","ca_his_raw_2017.h5","ca_his_raw_2018.h5","ca_his_raw_2019.h5") {
       kaggle datasets download liuxu77/largest -f $f -p data/largest --unzip
   }
   ```

2. Build the sensor table and the 5-minute flow file (about 10 minutes, mostly reading the HDF5 files; needs `h5py`, listed in `requirements.txt`):

   ```powershell
   python scripts/prepare_data.py --largest data/largest --out data
   ```

   This runs three steps: `quality` (share of missing and daytime-zero readings for every District 4 sensor, `d4_quality.csv`), `select` (the four corridors below, `sensors_100.csv`) and `extract` (`flow_5min_100.parquet`). `--steps select` alone reruns the selection in seconds when `d4_quality.csv` exists.

3. Download the weather (needs internet, a few minutes because of the API rate limit):

   ```powershell
   python scripts/fetch_weather.py
   ```

**Which sensors.** District 4 (Bay Area) sensors in the five core counties (Alameda, Contra Costa, San Francisco, San Mateo, Santa Clara). For every freeway direction with at least 25 sensors the script searches for 25 consecutive sensors, at most 3 km apart, each with at least 85% usable readings in both 2017 and 2018. The four corridors used are `C1_I580-E`, `C2_US101-N`, `C3_I880-S` and `C4_SR24-W` (25 sensors each). The first 20 sensors of each corridor are *warm* (in training), the last 5 are *cold* (only in the test rows). `sensors_100.csv` is reproduced exactly from `ca_meta.csv` and `d4_quality.csv`.

**Sources and licences**
* Traffic flow: [LargeST](https://github.com/liuxu77/LargeST) (Liu et al., NeurIPS 2023 Datasets and Benchmarks), California Bay Area subset, licensed CC BY-NC 4.0 (non-commercial use). Underlying data come from Caltrans PeMS.
* Weather: [Open-Meteo](https://open-meteo.com/) historical API (ERA5 reanalysis), fetched per corridor.

## How to reproduce

Run from the repository root with the virtual environment active. The original phase 2 and 3 plan is in `RUNBOOK_phase2_3.md` (it predates the analyses and the app, so follow this README where they differ).

```powershell
# 0. Data (see "Get the data" above)
python scripts/prepare_data.py --largest data/largest --out data
python scripts/fetch_weather.py

# 1. Build the 15-minute table and the feature tables
python -m src.cleaning
python -m src.features

# 2. Tests (leakage, parity)
python -m pytest tests -v -s

# 3. Notebooks, in order
#    01_eda -> 02_baselines_gate1 -> 03_fast_models

# 4. Slow models (Random Forest, AdaBoost, Gradient Boosting); can run overnight
.\scripts\run_slow.ps1

# 5. Validation analysis, then final refit (writes 2019 predictions once, refuses to overwrite)
#    notebooks/04_validation_analysis.ipynb
python scripts/run_model.py refit --models all --h 15 60 --seeds

# 6. Test scoring and freeze
#    notebooks/05_gate2_test.ipynb

# 7. Analyses
python scripts/run_ablation.py
python scripts/run_robustness.py
#    notebooks/06_analysis_E_errors.ipynb  (SPLIT = "val", then "test")

# 8. Extras: paired bootstrap, comparison with published results, tuning sensitivity (validation only, slow)
python scripts/paired_bootstrap.py
python scripts/compare_published.py
python scripts/run_tuning_sensitivity.py --h 15 60 --models lgbm xgb cat
python scripts/run_pca_study.py          # validation only, about 25 minutes

# 9. Report (needs a LaTeX installation). The generated tables are committed in report/tables/, so a plain
#    `cd report; pdflatex main.tex` (twice) works straight after cloning; make_tables.py regenerates them and
#    needs the feature tables built in steps 0 and 1.
python report/make_tables.py
cd report; pdflatex main.tex; pdflatex main.tex
```

Rules the code enforces: EDA and tuning use 2017 to 2018 only; 2019 is read only for the final refit and scoring; refit predictions are written once; the best baseline is chosen on validation and reused on test.

## Run the app

```powershell
python scripts/build_app_assets.py     # once: writes results/app/error_quantiles.json
streamlit run app/streamlit_app.py
```

The app's files are committed, so it runs straight after cloning (`pip install -r requirements.txt`, then the command above); `python -m pytest tests/test_app.py` runs five automated interaction tests (start-up, a fixed forecast, both horizons with the stress controls, every corridor, and the message shown when a data file is missing). Pick a sensor, a date and time, and a horizon. The app rebuilds the features from the flow history before that time only, predicts with LightGBM, and shows the forecast against recent flow and the baselines, an 80% interval taken from validation errors, the real value when the time is in the past, a corridor view, a per-prediction explanation, a stress test with noisy or missing readings, and the benchmark tables.

The app needs the trained LightGBM models in `models/` and the three data files above. The files it needs (`data/flow_15min_100.parquet`, `data/sensors_100.csv`, `data/weather_hourly.csv`, `models/lgbm_h15_final.joblib`, `models/lgbm_h60_final.joblib`, about 55 MB) are committed to this repository, so it runs right after cloning. If any of them is missing, the app lists what is missing and how to rebuild it instead of crashing.

## Limitations

* **Template sensors.** About 10% of sensor-days only repeat the previous week's values (they look like PeMS gap-filling with a weekly template). Eight sensors (2185 to 2190, 2806, 2814) are template on every day, so a model can score about 1% WAPE on them just by copying last week. The "clean rows" view drops sensor-days where at least 90% of the bins equal the same bin a week earlier or later (`src/quality.py`, rule fixed on 2017 to 2018 data). It keeps 87.2% of 2019 rows, and LightGBM reaches MAE 54.2 (15 min) and 67.2 (60 min) there, so the all-rows headline is flattering. Both views are in `results/final/headline_test_table.csv`.
* **Sensor dropouts.** Short runs of zero readings (under 2 hours) are kept as data, because they are what a live system would see. Most of the largest errors are simultaneous zeros across sensors that no feature can predict. Predictions can also be negative; they are not clipped in the reported results.
* **Tuning budget.** Searches were small (6 to 8 random draws for XGBoost and LightGBM), and several best settings sit on the edge of the grid (for example depth for XGBoost, CatBoost and Gradient Boosting, and the iteration cap for CatBoost at 2,000 and Gradient Boosting at 300 trees). The gaps among the top three models are small (about 0.1 to 0.3 MAE), so their order could change with wider tuning; the gap to the rest is large. A validation-only study (`results/tuning_sensitivity/summary.csv`) shows that a wider search would lower validation MAE by 0.3% (LightGBM) to 1.2% (CatBoost at 60 minutes), and at 60 minutes XGBoost would edge past LightGBM (61.76 vs 61.81). The frozen models were deliberately not changed after the test year was scored.
* **Weather** is hourly ERA5 reanalysis per corridor, not measured at the sensor, and in live use a real weather feed would be needed. It added no measurable accuracy (analysis A).
* **Cold sensors** have no rows in training, but their own recent flow history is still used as input (for the sensor scale and lags). They are about 0.6 WAPE points worse than warm sensors for every model, and 20 sensors is a small sample.

## Licence and attribution

Code in this repository is for coursework. LargeST data are CC BY-NC 4.0 and must be used non-commercially with attribution to the LargeST authors and Caltrans PeMS. Weather data are from Open-Meteo (ERA5).

## Team

| Name | Roll number |
|---|---|
| Praneeth Marudi | IMT2023555 |
| Vinay Kusumanchi | IMT2023608 |
| Shive S Bhat | BT2024067 |
| Mitansh | BT2024046 |
