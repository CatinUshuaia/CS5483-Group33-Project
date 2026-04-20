# CS5483 Group33 Project

China next-year life expectancy prediction project based on World Bank WDI indicators.

The repository currently contains:
- data preprocessing and fold-level dataset export
- baseline linear models (`LinearRegression`, `Ridge`, `Lasso`)
- advanced tree-based models (`RandomForestRegressor`, `XGBRegressor`)
- result plots and a browser-based comparison page

## Project Scope

- Data source: World Bank WDI API
- Country: `CHN`
- Target variable: `life_exp_next_year`
- Modeling sample: `1995-2022`
- Final test window: `2019-2022`
- Train/validation strategy: `TimeSeriesSplit(n_splits=4)` on years before `2019`

## Repository Structure

```text
CS5483-Group33-Project/
├── data_preprocessing/
│   ├── dataset/
│   │   ├── wdi_china_lifeexp_model_ready_no_clip.csv
│   │   └── processeddataset/
│   └── preprocessing_scripts/
│       ├── preprocess_china_lifeexp.py
│       └── split_time_series_datasets.py
├── baseline_models/
│   ├── train_baseline_models.py
│   └── results/
│       ├── report.md
│       ├── summary.json
│       └── ...
├── modeling/
│   ├── train_rf_models.py
│   ├── final_rf_test.py
│   ├── final_rf_test_clip.py
│   ├── train_xgboost_models.py
│   ├── final_xgboost_test.py
│   ├── plot_rf_results.py
│   ├── plot_xgboost_results.py
│   ├── modeling_README.md
│   └── outputs/
├── frontend.html
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Environment Setup

Recommended Python version: `3.10+`

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` already includes the packages used by preprocessing, baseline modeling, RF/XGBoost modeling, and plotting scripts.

## Quick Start

Run all commands from the repository root.

### 1. Build the base preprocessing dataset

```bash
python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py
```

Default output:
- `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv`

Optional example with explicit arguments:

```bash
python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py --country CHN --start-year 1995 --end-year 2023 --outdir data_preprocessing/dataset --details-path data_preprocessing/PREPROCESSING_DETAILS.md
```

### 2. Export fold-level datasets

```bash
python data_preprocessing/preprocessing_scripts/split_time_series_datasets.py --test-start-year 2019 --n-splits 4
```

Default behavior:
- Input base dataset: `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv`
- Train/validation period: `1995-2018`
- Final test period: `2019-2022`
- Output folder: `data_preprocessing/dataset/processeddataset/`

### 3. Train baseline models

```bash
python baseline_models/train_baseline_models.py
```

Main outputs are written to `baseline_models/results/`.

### 4. Train and evaluate advanced models

```bash
python modeling/train_rf_models.py
python modeling/train_xgboost_models.py
python modeling/final_rf_test.py
python modeling/final_rf_test_clip.py
python modeling/final_xgboost_test.py
```

Main outputs are written to `modeling/outputs/`.

### 5. Generate result plots

```bash
python modeling/plot_rf_results.py
python modeling/plot_xgboost_results.py
```

## Frontend Visualization

`frontend.html` provides a lightweight browser UI for comparing model outputs.

Supported models:
- Lasso
- Linear Regression
- Ridge
- Random Forest (`clip`)
- Random Forest (`no_clip`)
- XGBoost (`no_clip`)

Supported views:
- actual vs predicted
- coefficients or feature importance
- residual analysis
- summary metrics

Launch it with a local HTTP server instead of opening the file directly:

```bash
python -m http.server 8000
```

Then visit:

```text
http://localhost:8000/frontend.html
```

## Notes

- `data_preprocessing/PREPROCESSING_DETAILS.md` is not currently tracked in the repository. The preprocessing script can generate or update it via `--details-path`.
- `baseline_models/results/report.md` contains the current baseline experiment summary.
- `modeling/modeling_README.md` documents the RF/XGBoost workflow and outputs.

## License

This project is licensed under the MIT License. See `LICENSE`.
