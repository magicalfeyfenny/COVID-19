# COVID-19 Forecasting Project

This repo contains a Jupyter notebook workflow plus supporting scripts for the Johns Hopkins CSSE COVID-19 US time-series data.

Project layout:
- `notebooks/covid19_forecasting.ipynb` runs the baseline review and GRU benchmark.
- `scripts/prepare_dataset.py` prepares the state-level daily dataset.
- `scripts/process_pipeline.py` builds features, baselines, and sequence inputs.
- `scripts/gru_benchmark.py` contains the shared GRU benchmark helpers used by the notebook.
- `scripts/gru_direct.py` runs the original direct-target GRU configuration.
- `scripts/gru_residual.py` runs the tuned residual-target GRU configuration.
- `scripts/gru_state_consistency.py` runs the state-aware GRU configuration for stronger per-state consistency.
- `csse_covid_19_data/csse_covid_19_time_series/` contains the Johns Hopkins time-series CSVs already copied into the repo.
- `data/raw/` is an alternate location for local raw copies.
- `data/processed/` is for generated outputs.
- `figures/` is for exported plots.
- `requirements.txt` lists the notebook dependencies.

VSCode:
1. Use the repo venv at `.venv` or the `Python (.venv) COVID-19` notebook kernel.
2. Open `notebooks/covid19_forecasting.ipynb`.
3. Run the notebook top to bottom.

Optional CLI:
- `.venv/bin/python scripts/prepare_dataset.py`
- `.venv/bin/python scripts/process_pipeline.py`

Expected raw files:
- `time_series_covid19_confirmed_US.csv`
- `time_series_covid19_deaths_US.csv`
