# Modeling Module

This folder contains the modeling, testing, and visualization code for the **China life expectancy prediction** project.

Current models:
- Random Forest
- XGBoost

The default input data comes from:

`data_preprocessing/dataset/processeddataset/`

The target variable is:

`life_exp_next_year`

---

## Structure

```text
modeling/
├── train_rf_models.py
├── final_rf_test.py
├── final_rf_test_clip.py
├── plot_rf_results.py
├── train_xgboost_models.py
├── final_xgboost_test.py
├── plot_xgboost_results.py
└── outputs/
```

---

## Requirements

Recommended environment: **Python 3.10+**

Main dependencies:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

Install with:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

---

## Data

The scripts use preprocessed datasets with the following naming format:

- Training set: `wdi_china_lifeexp_model_ready_fold{K}_train_{version}.csv`
- Validation set: `wdi_china_lifeexp_model_ready_fold{K}_val_{version}.csv`
- Test set: `wdi_china_lifeexp_model_ready_test_{version}.csv`

Where:
- `K` = 1 to 4
- `version` = `clip` or `no_clip`

The label column is:
- `life_exp_next_year`

If a `year` column exists, it is excluded from training features but kept for result display.

---

## Usage

Run the following commands from the project root directory.

### 1. Random Forest parameter search
```bash
python modeling/train_rf_models.py
```

### 2. XGBoost parameter search
```bash
python modeling/train_xgboost_models.py
```

### 3. Final test
**RF (no_clip)**
```bash
python modeling/final_rf_test.py
```

**RF (clip)**
```bash
python modeling/final_rf_test_clip.py
```

**XGBoost (no_clip)**
```bash
python modeling/final_xgboost_test.py
```

### 4. Visualization
```bash
python modeling/plot_rf_results.py
python modeling/plot_xgboost_results.py
```

---

## Outputs

The `outputs/` folder may contain:

- prediction results on the test set
- feature importance tables
- true vs. predicted plots
- feature importance plots

---

## Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

Lower values indicate better performance.

---

## Notes

- Please run scripts from the project root directory.
- Make sure the required `clip` / `no_clip` dataset files exist before running.
- In headless environments, you may need to disable `plt.show()` or use a non-interactive matplotlib backend.
