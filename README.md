# CS5483 Group33 Project

China next-year life expectancy prediction project based on World Bank WDI indicators.

This repository includes:
- data preprocessing and fold-level dataset export
- baseline linear models (Linear Regression / Ridge / Lasso)
- advanced models (Random Forest / XGBoost)
- frontend visualization for model result comparison

## Project Scope

- Data source: World Bank WDI API
- Target variable: `life_exp_next_year`
- Country: `CHN`
- Time range used in modeling: `1995-2022` (label uses next year)
- Time-series split setting: train/validation on years before `2019`, final test on `2019-2022`

## Repository Structure

```text
CS5483-Group33-Project/
├── data_preprocessing/
│   ├── dataset/
│   │   ├── wdi_china_lifeexp_model_ready_no_clip.csv
│   │   └── processeddataset/
│   ├── preprocessing_scripts/
│   │   ├── preprocess_china_lifeexp.py
│   │   └── split_time_series_datasets.py
│   ├── PREPROCESSING_DETAILS.md
│   └── FEATURE_SELECTION_LITERATURE_NOTE.md
├── baseline_models/
│   ├── train_baseline_models.py
│   └── results/
├── modeling/
│   ├── train_rf_models.py
│   ├── final_rf_test.py
│   ├── final_rf_test_clip.py
│   ├── train_xgboost_models.py
│   ├── final_xgboost_test.py
│   ├── plot_rf_results.py
│   ├── plot_xgboost_results.py
│   └── modeling_README.md
├── frontend.html
├── requirements.txt
└── LICENSE
```

## Environment Setup

Recommended Python version: `3.10+`

Install base dependencies:

```bash
pip install -r requirements.txt
```

For modeling scripts, make sure additional packages used by those scripts are installed (for example `numpy` and `xgboost`). See `modeling/modeling_README.md` for details.

## Quick Start (End-to-End)

Run all commands from the repository root.

### 1) Build base preprocessing dataset

```bash
python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py
```

### 2) Export fold-level datasets with `TimeSeriesSplit`

```bash
python data_preprocessing/preprocessing_scripts/split_time_series_datasets.py --test-start-year 2019 --n-splits 4
```

Default behavior:
- Input base dataset: `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv`
- Train/validation period: years before `2019`
- Final test period: `2019-2022`
- Output folder: `data_preprocessing/dataset/processeddataset/`

### 3) Train baseline models (Linear / Ridge / Lasso)

```bash
python baseline_models/train_baseline_models.py
```

### 4) Train and evaluate advanced models (RF / XGBoost)

```bash
python modeling/train_rf_models.py
python modeling/train_xgboost_models.py
python modeling/final_rf_test.py
python modeling/final_rf_test_clip.py
python modeling/final_xgboost_test.py
```

### 5) Generate result plots

```bash
python modeling/plot_rf_results.py
python modeling/plot_xgboost_results.py
```

## Frontend Visualization

The project includes a browser-based page (`frontend.html`) for comparing model outputs.

### Supported Visualizations

- Actual vs Predicted line chart
- Top coefficients for linear models
- Feature importance for tree-based models
- Residuals on test set
- Interactive model switching in browser

### Supported Models

- Linear Regression
- Ridge
- Lasso
- Random Forest (clip)
- Random Forest (no clip)
- XGBoost (no clip)

### How to Launch

Do not open `frontend.html` directly by double-clicking. Start a local HTTP server from the project root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/frontend.html
```

## Additional Documentation

- Preprocessing details: `data_preprocessing/PREPROCESSING_DETAILS.md`
- Modeling module usage and results: `modeling/modeling_README.md`
- Baseline experiment report: `baseline_models/results/report.md`

## License

This project is licensed under the MIT License. See `LICENSE`.
