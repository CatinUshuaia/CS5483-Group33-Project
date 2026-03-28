from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data_preprocessing" / "dataset" / "processeddataset"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_xy(csv_path, target_col="life_exp_next_year"):
    df = pd.read_csv(csv_path)
    drop_cols = [target_col]
    if "year" in df.columns:
        drop_cols.append("year")
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


def evaluate_one_fold(train_path, val_path, params):
    X_train, y_train = load_xy(train_path)
    X_val, y_val = load_xy(val_path)

    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return {
        "mae": mean_absolute_error(y_val, pred),
        "rmse": rmse(y_val, pred)
    }


def evaluate_4fold(version="no_clip", params=None):
    if params is None:
        params = {
            "n_estimators": 50,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }

    results = []

    for fold in [1, 2, 3, 4]:
        train_path = DATA_DIR / f"wdi_china_lifeexp_model_ready_fold{fold}_train_{version}.csv"
        val_path = DATA_DIR / f"wdi_china_lifeexp_model_ready_fold{fold}_val_{version}.csv"

        fold_result = evaluate_one_fold(train_path, val_path, params)
        fold_result["fold"] = fold
        results.append(fold_result)

    results_df = pd.DataFrame(results)
    print(f"\n=== XGBoost | version={version} ===")
    print(results_df)
    print("\nAverage:")
    print(results_df[["mae", "rmse"]].mean())

    return results_df


def run_param_search(version="clip"):
    param_grid = [
        {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    ]

    all_results = []

    for params in param_grid:
        results_df = evaluate_4fold(version=version, params=params)
        avg_mae = results_df["mae"].mean()
        avg_rmse = results_df["rmse"].mean()

        row = params.copy()
        row["version"] = version
        row["avg_mae"] = avg_mae
        row["avg_rmse"] = avg_rmse
        all_results.append(row)

    summary_df = pd.DataFrame(all_results).sort_values(by="avg_mae")
    print("\n=== Parameter Search Summary ===")
    print(summary_df)
    return summary_df


def main():
    summary_clip = run_param_search(version="clip")
    summary_no_clip = run_param_search(version="no_clip")

    print("\n=== Best clip setting ===")
    print(summary_clip.iloc[0])

    print("\n=== Best no_clip setting ===")
    print(summary_no_clip.iloc[0])


if __name__ == "__main__":
    main()