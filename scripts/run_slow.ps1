# Runs the slow models one after another (safe to stop and restart; finished pairs are skipped).
# From the repo root, venv active:   .\scripts\run_slow.ps1
python scripts/run_model.py tune --models rf --h 15 60
python scripts/run_model.py tune --models ada --h 15 60
python scripts/run_model.py tune --models gb --h 15 60
Write-Host "Slow models finished."
