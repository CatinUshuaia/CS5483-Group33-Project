"""
Advanced Models: Random Forest & XGBoost for China Life Expectancy Prediction.

Uses the pre-split TimeSeriesSplit fold CSVs produced by
data_preprocessing/preprocessing_scripts/split_time_series_datasets.py.

Workflow
--------
1. Load fold-level train/val CSVs (4 folds).
2. Grid-search over hyperparameters using the pre-built folds as CV.
3. Select the best hyperparameter combo by mean validation RMSE.
4. Retrain on the full train+val period (1995-2018) and evaluate on test (2019-2022).
5. Save results, feature importances, and plots.
"""

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "life_exp_next_year"
EXCLUDE_COLS = {"year", TARGET}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fold_data(
    data_dir: Path,
    stem: str,
    variant: str,
    n_folds: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Load pre-split fold train/val pairs."""
    folds = []
    for i in range(1, n_folds + 1):
        train_path = data_dir / f"{stem}_fold{i}_train_{variant}.csv"
        val_path = data_dir / f"{stem}_fold{i}_val_{variant}.csv"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Missing fold files: {train_path.name} or {val_path.name}"
            )
        folds.append((pd.read_csv(train_path), pd.read_csv(val_path)))
    return folds


def load_test_data(data_dir: Path, stem: str, variant: str) -> pd.DataFrame:
    """Load the held-out test set."""
    test_path = data_dir / f"{stem}_test_{variant}.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path.name}")
    return pd.read_csv(test_path)


def load_base_dataset(base_csv: Path, test_start_year: int):
    """Load base dataset and split into train_val / test for final retraining."""
    df = pd.read_csv(base_csv).sort_values("year").reset_index(drop=True)
    train_val = df[df["year"] < test_start_year].copy()
    test = df[df["year"] >= test_start_year].copy()
    return train_val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE(%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

RF_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5, 1.0],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 5],
}


def _grid_dicts(grid: dict) -> list[dict]:
    """Expand a param grid into a list of individual param combos."""
    keys = list(grid.keys())
    vals = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ---------------------------------------------------------------------------
# Cross-validation with pre-split folds
# ---------------------------------------------------------------------------

def cv_evaluate(
    model_cls,
    params: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
) -> tuple[float, list[float]]:
    """Train on each fold's train set, evaluate on val; return mean RMSE & per-fold RMSEs."""
    fold_rmses = []
    for train_df, val_df in folds:
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET].values

        model = model_cls(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        fold_rmses.append(rmse)
    return float(np.mean(fold_rmses)), fold_rmses


def grid_search_cv(
    model_name: str,
    model_cls,
    param_grid: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
) -> tuple[dict, float, list[float]]:
    """Exhaustive grid search over pre-split folds. Returns best params, best mean RMSE, fold RMSEs."""
    combos = _grid_dicts(param_grid)
    total = len(combos)
    print(f"\n[{model_name}] Grid search: {total} combinations x {len(folds)} folds")

    best_rmse = float("inf")
    best_params: dict = {}
    best_fold_rmses: list[float] = []

    for idx, params in enumerate(combos, 1):
        mean_rmse, fold_rmses = cv_evaluate(model_cls, params, folds, feature_cols)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params
            best_fold_rmses = fold_rmses
        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] current best RMSE = {best_rmse:.4f}")

    print(f"  Best params: {best_params}")
    print(f"  Best mean val RMSE: {best_rmse:.4f}  (folds: {[round(r, 4) for r in best_fold_rmses]})")
    return best_params, best_rmse, best_fold_rmses


# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------

def train_and_evaluate_final(
    model_name: str,
    model_cls,
    best_params: dict,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> dict:
    """Retrain with best params on full train+val, evaluate on test, save artifacts."""
    X_train = train_val_df[feature_cols].values
    y_train = train_val_df[TARGET].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET].values

    model = model_cls(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    print(f"\n[{model_name}] Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    importances = _get_feature_importances(model, feature_cols)

    predictions_df = pd.DataFrame({
        "year": test_df["year"].values,
        "actual": y_test,
        "predicted": y_pred,
        "residual": y_test - y_pred,
    })

    model_dir = output_dir / model_name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    predictions_df.to_csv(model_dir / "test_predictions.csv", index=False)

    results = {
        "model": model_name,
        "best_params": _serialize_params(best_params),
        "test_metrics": metrics,
        "feature_importances": importances,
    }
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_actual_vs_predicted(predictions_df, model_name, model_dir)
    _plot_residuals(predictions_df, model_name, model_dir)
    _plot_feature_importance(importances, model_name, model_dir)

    return results


def _get_feature_importances(model, feature_cols: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return {}
    return {col: float(v) for col, v in zip(feature_cols, imp)}


def _serialize_params(params: dict) -> dict:
    """Make params JSON-serializable."""
    out = {}
    for k, v in params.items():
        if v is None:
            out[k] = "None"
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_actual_vs_predicted(pred_df: pd.DataFrame, model_name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pred_df["year"], pred_df["actual"], "o-", label="Actual", linewidth=2)
    ax.plot(pred_df["year"], pred_df["predicted"], "s--", label="Predicted", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Life Expectancy (next year)")
    ax.set_title(f"{model_name} – Actual vs Predicted (Test 2019-2022)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def _plot_residuals(pred_df: pd.DataFrame, model_name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pred_df["year"].astype(str), pred_df["residual"], color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{model_name} – Residuals on Test Set")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(importances: dict, model_name: str, out_dir: Path, top_n: int = 15):
    if not importances:
        return
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), values, color="darkorange", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} – Top {top_n} Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(all_results: list[dict], output_dir: Path):
    """Side-by-side metric comparison across all models."""
    if len(all_results) < 2:
        return

    models = [r["model"] for r in all_results]
    metric_names = ["RMSE", "MAE", "R2", "MAPE(%)"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    for ax, metric in zip(axes, metric_names):
        vals = [r["test_metrics"][metric] for r in all_results]
        bars = ax.bar(models, vals, color=["steelblue", "darkorange"], alpha=0.85)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison on Test Set (2019-2022)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest & XGBoost for life expectancy prediction."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_preprocessing/dataset/processeddataset",
        help="Directory with fold-level CSV files.",
    )
    parser.add_argument(
        "--base-csv",
        type=str,
        default="data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv",
        help="Base (unprocessed) CSV for final retraining.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="wdi_china_lifeexp_model_ready",
        help="File name stem for fold CSVs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="no_clip",
        choices=["no_clip", "clip", "no_clip_scaled", "clip_scaled"],
        help="Data variant to use for CV folds (tree models don't need scaling).",
    )
    parser.add_argument("--n-folds", type=int, default=4, help="Number of folds.")
    parser.add_argument(
        "--test-start-year", type=int, default=2019, help="First year of test set."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="advanced_models/results",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced param grid for faster iteration.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_csv = Path(args.base_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Advanced Models: Random Forest & XGBoost")
    print("=" * 60)
    print(f"  Data dir    : {data_dir}")
    print(f"  Variant     : {args.variant}")
    print(f"  Folds       : {args.n_folds}")
    print(f"  Output dir  : {output_dir}")

    folds = load_fold_data(data_dir, args.stem, args.variant, args.n_folds)
    feature_cols = get_feature_cols(folds[0][0])
    print(f"  Features    : {len(feature_cols)}")
    print(f"  Fold sizes  : {[(len(t), len(v)) for t, v in folds]}")

    train_val_df, test_df = load_base_dataset(base_csv, args.test_start_year)
    print(f"  Train+Val   : {len(train_val_df)} rows ({int(train_val_df['year'].min())}-{int(train_val_df['year'].max())})")
    print(f"  Test        : {len(test_df)} rows ({int(test_df['year'].min())}-{int(test_df['year'].max())})")

    rf_grid = RF_PARAM_GRID
    xgb_grid = XGB_PARAM_GRID
    if args.quick:
        rf_grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, None],
            "min_samples_split": [2],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 1.0],
        }
        xgb_grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "reg_alpha": [0],
            "reg_lambda": [1],
        }

    all_results = []

    # --- Random Forest ---
    rf_best_params, rf_cv_rmse, rf_fold_rmses = grid_search_cv(
        "Random Forest",
        RandomForestRegressor,
        rf_grid,
        folds,
        feature_cols,
    )
    rf_best_params["random_state"] = 42
    rf_best_params["n_jobs"] = -1
    rf_results = train_and_evaluate_final(
        "Random Forest",
        RandomForestRegressor,
        rf_best_params,
        train_val_df,
        test_df,
        feature_cols,
        output_dir,
    )
    rf_results["cv_mean_rmse"] = rf_cv_rmse
    rf_results["cv_fold_rmses"] = rf_fold_rmses
    all_results.append(rf_results)

    # --- XGBoost ---
    xgb_base_params = {"objective": "reg:squarederror", "random_state": 42, "verbosity": 0}
    xgb_best_params, xgb_cv_rmse, xgb_fold_rmses = grid_search_cv(
        "XGBoost",
        XGBRegressor,
        xgb_grid,
        folds,
        feature_cols,
    )
    xgb_best_params.update(xgb_base_params)
    xgb_results = train_and_evaluate_final(
        "XGBoost",
        XGBRegressor,
        xgb_best_params,
        train_val_df,
        test_df,
        feature_cols,
        output_dir,
    )
    xgb_results["cv_mean_rmse"] = xgb_cv_rmse
    xgb_results["cv_fold_rmses"] = xgb_fold_rmses
    all_results.append(xgb_results)

    # --- Comparison ---
    plot_model_comparison(all_results, output_dir)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in all_results:
        m = r["test_metrics"]
        print(f"  {r['model']:<15}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}  MAPE={m['MAPE(%)']:.2f}%")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
