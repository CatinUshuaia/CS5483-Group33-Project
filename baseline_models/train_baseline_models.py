"""
Baseline Models: Linear Regression / Ridge / Lasso for China Life Expectancy Prediction.

Uses pre-split TimeSeriesSplit fold CSVs (4 folds) for cross-validation
and a held-out test set (2019-2022) for final evaluation.

Workflow
--------
1. Load fold-level train/val CSVs (clip_scaled variant — linear models
   benefit from standardised, outlier-clipped features).
2. For Ridge and Lasso, sweep regularisation strength (alpha) over folds.
3. Train final models on full train+val period, evaluate on test set.
4. Save metrics, coefficients, predictions, and plots.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "life_exp_next_year"
EXCLUDE_COLS = {"year", TARGET}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fold_data(
    data_dir: Path, stem: str, variant: str, n_folds: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    folds = []
    for i in range(1, n_folds + 1):
        tp = data_dir / f"{stem}_fold{i}_train_{variant}.csv"
        vp = data_dir / f"{stem}_fold{i}_val_{variant}.csv"
        if not tp.exists() or not vp.exists():
            raise FileNotFoundError(f"Missing: {tp.name} or {vp.name}")
        folds.append((pd.read_csv(tp), pd.read_csv(vp)))
    return folds


def load_test_data(data_dir: Path, stem: str, variant: str) -> pd.DataFrame:
    tp = data_dir / f"{stem}_test_{variant}.csv"
    if not tp.exists():
        raise FileNotFoundError(f"Missing: {tp.name}")
    return pd.read_csv(tp)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE(%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


# ---------------------------------------------------------------------------
# Cross-validation alpha sweep
# ---------------------------------------------------------------------------

ALPHA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def cv_evaluate(
    model_cls,
    params: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
) -> tuple[float, list[float]]:
    fold_rmses = []
    for train_df, val_df in folds:
        X_tr = train_df[feature_cols].values
        y_tr = train_df[TARGET].values
        X_va = val_df[feature_cols].values
        y_va = val_df[TARGET].values

        model = model_cls(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        fold_rmses.append(float(np.sqrt(mean_squared_error(y_va, y_pred))))
    return float(np.mean(fold_rmses)), fold_rmses


def sweep_alpha(
    model_name: str,
    model_cls,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    extra_params: dict | None = None,
) -> tuple[dict, float, list[float], list[tuple[float, float]]]:
    """Sweep alpha values and return best params, best RMSE, fold RMSEs, and full curve."""
    extra_params = extra_params or {}
    print(f"\n[{model_name}] Alpha sweep: {len(ALPHA_GRID)} values x {len(folds)} folds")

    best_rmse = float("inf")
    best_alpha = ALPHA_GRID[0]
    best_fold_rmses: list[float] = []
    curve: list[tuple[float, float]] = []

    for alpha in ALPHA_GRID:
        params = {"alpha": alpha, **extra_params}
        mean_rmse, fold_rmses = cv_evaluate(model_cls, params, folds, feature_cols)
        curve.append((alpha, mean_rmse))
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_alpha = alpha
            best_fold_rmses = fold_rmses

    print(f"  Best alpha = {best_alpha}  →  mean val RMSE = {best_rmse:.4f}")
    print(f"  Fold RMSEs: {[round(r, 4) for r in best_fold_rmses]}")

    best_params = {"alpha": best_alpha, **extra_params}
    return best_params, best_rmse, best_fold_rmses, curve


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_final(
    model_name: str,
    model_cls,
    params: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    test_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> dict:
    all_train = pd.concat([t for t, _ in folds] + [v for _, v in folds])
    X_train = all_train[feature_cols].values
    y_train = all_train[TARGET].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET].values

    model = model_cls(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    print(f"\n[{model_name}] Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    coef_dict = _get_coefficients(model, feature_cols)

    predictions_df = pd.DataFrame({
        "year": test_df["year"].values,
        "actual": y_test,
        "predicted": y_pred,
        "residual": y_test - y_pred,
    })

    model_dir = output_dir / model_name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(model_dir / "test_predictions.csv", index=False)

    results = {
        "model": model_name,
        "params": _serialize(params),
        "test_metrics": metrics,
        "coefficients": coef_dict,
    }
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_actual_vs_predicted(predictions_df, model_name, model_dir)
    _plot_residuals(predictions_df, model_name, model_dir)
    if coef_dict:
        _plot_coefficients(coef_dict, model_name, model_dir)

    return results


def _get_coefficients(model, feature_cols: list[str]) -> dict:
    if not hasattr(model, "coef_"):
        return {}
    coefs = {"intercept": float(model.intercept_)}
    for col, val in zip(feature_cols, model.coef_):
        coefs[col] = float(val)
    return coefs


def _serialize(params: dict) -> dict:
    return {k: (str(v) if v is None else v) for k, v in params.items()}


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
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"{model_name} – Residuals on Test Set")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)


def _plot_coefficients(coefs: dict, model_name: str, out_dir: Path, top_n: int = 15):
    items = {k: v for k, v in coefs.items() if k != "intercept"}
    sorted_items = sorted(items.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [s[0] for s in sorted_items]
    values = [s[1] for s in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in values]
    ax.barh(range(len(names)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient Value")
    ax.set_title(f"{model_name} – Top {top_n} Coefficients (magnitude)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "coefficients.png", dpi=150)
    plt.close(fig)


def plot_alpha_curves(curves: dict[str, list[tuple[float, float]]], output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in curves.items():
        alphas = [c[0] for c in curve]
        rmses = [c[1] for c in curve]
        ax.plot(alphas, rmses, "o-", label=name, linewidth=2, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (regularisation strength, log scale)")
    ax.set_ylabel("Mean Validation RMSE")
    ax.set_title("Ridge / Lasso – Alpha Tuning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_tuning_curves.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(all_results: list[dict], output_dir: Path):
    if len(all_results) < 2:
        return
    models = [r["model"] for r in all_results]
    metric_names = ["RMSE", "MAE", "R2", "MAPE(%)"]
    colors = ["#5cb85c", "#5bc0de", "#f0ad4e"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    for ax, metric in zip(axes, metric_names):
        vals = [r["test_metrics"][metric] for r in all_results]
        bars = ax.bar(models, vals, color=colors[: len(models)], alpha=0.85)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Baseline Model Comparison on Test Set (2019-2022)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coef_heatmap(all_results: list[dict], feature_cols: list[str], output_dir: Path):
    """Heatmap of coefficients across Linear/Ridge/Lasso for comparison."""
    models_with_coefs = [r for r in all_results if r.get("coefficients")]
    if len(models_with_coefs) < 2:
        return

    sorted_feats = sorted(
        feature_cols,
        key=lambda f: abs(models_with_coefs[0]["coefficients"].get(f, 0)),
        reverse=True,
    )[:15]

    data = []
    model_names = []
    for r in models_with_coefs:
        model_names.append(r["model"])
        data.append([r["coefficients"].get(f, 0) for f in sorted_feats])

    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(arr)), vmax=np.max(np.abs(arr)))
    ax.set_xticks(range(len(sorted_feats)))
    ax.set_xticklabels(sorted_feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_title("Coefficient Comparison: Linear vs Ridge vs Lasso (top 15 by magnitude)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "coefficient_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Linear/Ridge/Lasso baseline models for life expectancy prediction."
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="data_preprocessing/dataset/processeddataset",
    )
    parser.add_argument("--stem", type=str, default="wdi_china_lifeexp_model_ready")
    parser.add_argument(
        "--variant", type=str, default="clip_scaled",
        choices=["no_clip", "clip", "no_clip_scaled", "clip_scaled"],
        help="Data variant. clip_scaled recommended for linear models.",
    )
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=str, default="baseline_models/results",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Baseline Models: Linear / Ridge / Lasso")
    print("=" * 60)
    print(f"  Data dir : {data_dir}")
    print(f"  Variant  : {args.variant}")
    print(f"  Folds    : {args.n_folds}")
    print(f"  Output   : {output_dir}")

    folds = load_fold_data(data_dir, args.stem, args.variant, args.n_folds)
    test_df = load_test_data(data_dir, args.stem, args.variant)
    feature_cols = get_feature_cols(folds[0][0])
    print(f"  Features : {len(feature_cols)}")
    print(f"  Fold sizes : {[(len(t), len(v)) for t, v in folds]}")
    print(f"  Test size  : {len(test_df)}")

    all_results = []
    alpha_curves: dict[str, list[tuple[float, float]]] = {}

    # --- 1. Linear Regression (no regularisation) ---
    print("\n" + "-" * 40)
    print("1. Linear Regression (OLS)")
    print("-" * 40)
    lr_rmse, lr_fold_rmses = cv_evaluate(
        LinearRegression, {}, folds, feature_cols,
    )
    print(f"  CV mean RMSE: {lr_rmse:.4f}  (folds: {[round(r, 4) for r in lr_fold_rmses]})")
    lr_results = train_and_evaluate_final(
        "Linear Regression", LinearRegression, {},
        folds, test_df, feature_cols, output_dir,
    )
    lr_results["cv_mean_rmse"] = lr_rmse
    lr_results["cv_fold_rmses"] = lr_fold_rmses
    all_results.append(lr_results)

    # --- 2. Ridge Regression ---
    print("\n" + "-" * 40)
    print("2. Ridge Regression")
    print("-" * 40)
    ridge_params, ridge_cv_rmse, ridge_fold_rmses, ridge_curve = sweep_alpha(
        "Ridge", Ridge, folds, feature_cols,
    )
    alpha_curves["Ridge"] = ridge_curve
    ridge_results = train_and_evaluate_final(
        "Ridge", Ridge, ridge_params,
        folds, test_df, feature_cols, output_dir,
    )
    ridge_results["cv_mean_rmse"] = ridge_cv_rmse
    ridge_results["cv_fold_rmses"] = ridge_fold_rmses
    all_results.append(ridge_results)

    # --- 3. Lasso Regression ---
    print("\n" + "-" * 40)
    print("3. Lasso Regression")
    print("-" * 40)
    lasso_params, lasso_cv_rmse, lasso_fold_rmses, lasso_curve = sweep_alpha(
        "Lasso", Lasso, folds, feature_cols, extra_params={"max_iter": 10000},
    )
    alpha_curves["Lasso"] = lasso_curve
    lasso_results = train_and_evaluate_final(
        "Lasso", Lasso, {**lasso_params, "max_iter": 10000},
        folds, test_df, feature_cols, output_dir,
    )
    lasso_results["cv_mean_rmse"] = lasso_cv_rmse
    lasso_results["cv_fold_rmses"] = lasso_fold_rmses
    all_results.append(lasso_results)

    # --- Comparison plots ---
    plot_alpha_curves(alpha_curves, output_dir)
    plot_model_comparison(all_results, output_dir)
    plot_coef_heatmap(all_results, feature_cols, output_dir)

    # --- Summary ---
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in all_results:
        m = r["test_metrics"]
        alpha_str = ""
        if "alpha" in r.get("params", {}):
            alpha_str = f"  alpha={r['params']['alpha']}"
        print(
            f"  {r['model']:<20}{alpha_str:<12} "
            f"RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
            f"R²={m['R2']:.4f}  MAPE={m['MAPE(%)']:.2f}%"
        )

    # Lasso sparsity analysis
    lasso_coefs = lasso_results.get("coefficients", {})
    if lasso_coefs:
        non_zero = sum(1 for k, v in lasso_coefs.items() if k != "intercept" and abs(v) > 1e-8)
        total = len([k for k in lasso_coefs if k != "intercept"])
        print(f"\n  Lasso sparsity: {non_zero}/{total} non-zero coefficients")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
