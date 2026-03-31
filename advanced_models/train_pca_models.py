"""
PCA-enhanced Advanced Models for China Life Expectancy Prediction.

Addresses the curse of dimensionality identified in the baseline tree experiments
by applying StandardScaler + PCA before training Random Forest / XGBoost.

Experiments run:
  A) PCA on all 31 features (with life_exp)
  B) PCA on 30 features (without life_exp) — forces multi-indicator learning

For each experiment, PCA is fit on fold-training data only to prevent leakage.
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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "life_exp_next_year"
EXCLUDE_COLS = {"year", TARGET}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_base_dataset(base_csv: Path, test_start_year: int):
    df = pd.read_csv(base_csv).sort_values("year").reset_index(drop=True)
    train_val = df[df["year"] < test_start_year].copy()
    test = df[df["year"] >= test_start_year].copy()
    return train_val, test


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


def get_feature_cols(df: pd.DataFrame, drop_life_exp: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if drop_life_exp:
        cols = [c for c in cols if c != "life_exp"]
    return cols


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
# PCA analysis
# ---------------------------------------------------------------------------

def analyse_pca_variance(X: np.ndarray, feature_cols: list[str], output_dir: Path, label: str):
    """Fit PCA on full feature set and plot cumulative explained variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA().fit(X_scaled)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cum_var, 0.95) + 1)
    n_99 = int(np.searchsorted(cum_var, 0.99) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, "o-", markersize=4)
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.7, label=f"95% → {n_95} PCs")
    ax.axhline(0.99, color="orange", linestyle="--", alpha=0.7, label=f"99% → {n_99} PCs")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"PCA Cumulative Variance ({label})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"pca_variance_{label}.png", dpi=150)
    plt.close(fig)

    info = {
        "n_features": len(feature_cols),
        "n_components_95pct": n_95,
        "n_components_99pct": n_99,
        "explained_variance_ratio": pca_full.explained_variance_ratio_.tolist(),
        "cumulative_variance": cum_var.tolist(),
    }
    return info


# ---------------------------------------------------------------------------
# Hyperparameter grids (compact — PCA reduces complexity needs)
# ---------------------------------------------------------------------------

RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [2, 3, 5, None],
    "min_samples_leaf": [1, 2, 3],
}

XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 1.0],
    "reg_lambda": [1, 5],
}


def _grid_dicts(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    vals = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ---------------------------------------------------------------------------
# CV with per-fold PCA fitting
# ---------------------------------------------------------------------------

def cv_evaluate_pca(
    model_cls,
    params: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    n_components: int | float,
) -> tuple[float, list[float]]:
    fold_rmses = []
    for train_df, val_df in folds:
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
        ])
        X_train_pca = pipe.fit_transform(X_train)
        X_val_pca = pipe.transform(X_val)

        model = model_cls(**params)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_val_pca)
        fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

    return float(np.mean(fold_rmses)), fold_rmses


def grid_search_pca(
    model_name: str,
    model_cls,
    param_grid: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    n_components: int | float,
) -> tuple[dict, float, list[float]]:
    combos = _grid_dicts(param_grid)
    total = len(combos)
    print(f"\n  [{model_name}] Grid search: {total} combos x {len(folds)} folds (PCA n_components={n_components})")

    best_rmse = float("inf")
    best_params: dict = {}
    best_fold_rmses: list[float] = []

    for idx, params in enumerate(combos, 1):
        mean_rmse, fold_rmses = cv_evaluate_pca(
            model_cls, params, folds, feature_cols, n_components
        )
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params
            best_fold_rmses = fold_rmses
        if idx % 50 == 0 or idx == total:
            print(f"    [{idx}/{total}] best RMSE = {best_rmse:.4f}")

    print(f"    Best params: {best_params}")
    print(f"    Best mean val RMSE: {best_rmse:.4f}")
    return best_params, best_rmse, best_fold_rmses


# ---------------------------------------------------------------------------
# Final train + evaluate
# ---------------------------------------------------------------------------

def train_final_pca(
    model_name: str,
    model_cls,
    best_params: dict,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int | float,
    output_dir: Path,
    tag: str,
) -> dict:
    X_train = train_val_df[feature_cols].values
    y_train = train_val_df[TARGET].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
    ])
    X_train_pca = pipe.fit_transform(X_train)
    X_test_pca = pipe.transform(X_test)

    actual_n = X_train_pca.shape[1]

    model = model_cls(**best_params)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    metrics = compute_metrics(y_test, y_pred)
    print(f"\n  [{model_name} | {tag}] Test metrics (PCA {actual_n} components):")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    predictions_df = pd.DataFrame({
        "year": test_df["year"].values,
        "actual": y_test,
        "predicted": y_pred,
        "residual": y_test - y_pred,
    })

    safe_name = f"{model_name.lower().replace(' ', '_')}_{tag}"
    model_dir = output_dir / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"pipeline": pipe, "model": model}, model_dir / "model.joblib")
    predictions_df.to_csv(model_dir / "test_predictions.csv", index=False)

    importances = {}
    if hasattr(model, "feature_importances_"):
        pc_names = [f"PC{i+1}" for i in range(actual_n)]
        importances = {n: float(v) for n, v in zip(pc_names, model.feature_importances_)}

    _plot_actual_vs_predicted(predictions_df, f"{model_name} ({tag})", model_dir)
    _plot_residuals(predictions_df, f"{model_name} ({tag})", model_dir)
    if importances:
        _plot_pc_importance(importances, f"{model_name} ({tag})", model_dir)

    results = {
        "model": model_name,
        "tag": tag,
        "n_components_requested": n_components if isinstance(n_components, int) else str(n_components),
        "n_components_actual": actual_n,
        "best_params": _serialize(best_params),
        "test_metrics": metrics,
        "pc_importances": importances,
    }
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _serialize(params: dict) -> dict:
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

def _plot_actual_vs_predicted(pred_df, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pred_df["year"], pred_df["actual"], "o-", label="Actual", linewidth=2)
    ax.plot(pred_df["year"], pred_df["predicted"], "s--", label="Predicted", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Life Expectancy (next year)")
    ax.set_title(f"{model_name} – Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def _plot_residuals(pred_df, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pred_df["year"].astype(str), pred_df["residual"], color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"{model_name} – Residuals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)


def _plot_pc_importance(importances, model_name, out_dir):
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(range(len(names)), values, color="darkorange", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} – PC Importances")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "pc_importance.png", dpi=150)
    plt.close(fig)


def plot_all_comparison(all_results: list[dict], output_dir: Path):
    if len(all_results) < 2:
        return
    labels = [f"{r['model']}\n({r['tag']})" for r in all_results]
    metric_names = ["RMSE", "MAE", "R2", "MAPE(%)"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
    for ax, metric in zip(axes, metric_names):
        vals = [r["test_metrics"][metric] for r in all_results]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("PCA Experiment: Model Comparison on Test Set", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA-enhanced RF & XGBoost for life expectancy prediction."
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="data_preprocessing/dataset/processeddataset",
    )
    parser.add_argument(
        "--base-csv", type=str,
        default="data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv",
    )
    parser.add_argument("--stem", type=str, default="wdi_china_lifeexp_model_ready")
    parser.add_argument("--variant", type=str, default="no_clip")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--test-start-year", type=int, default=2019)
    parser.add_argument(
        "--output-dir", type=str, default="advanced_models/results_pca",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_csv = Path(args.base_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PCA-Enhanced Advanced Models")
    print("=" * 60)

    folds = load_fold_data(data_dir, args.stem, args.variant, args.n_folds)
    train_val_df, test_df = load_base_dataset(base_csv, args.test_start_year)

    all_results = []

    # =======================================================================
    # Experiment A: PCA on ALL 31 features (including life_exp)
    # =======================================================================
    print("\n>>> Experiment A: PCA on all features (with life_exp)")
    feat_all = get_feature_cols(folds[0][0], drop_life_exp=False)
    print(f"  Features: {len(feat_all)}")

    pca_info_all = analyse_pca_variance(
        train_val_df[feat_all].values, feat_all, output_dir, "all_features"
    )
    n_comp_all = pca_info_all["n_components_95pct"]
    print(f"  PCA 95% variance → {n_comp_all} components (from {len(feat_all)})")

    for model_name, model_cls, grid, extra in [
        ("Random Forest", RandomForestRegressor, RF_GRID, {"random_state": 42, "n_jobs": -1}),
        ("XGBoost", XGBRegressor, XGB_GRID, {"objective": "reg:squarederror", "random_state": 42, "verbosity": 0}),
    ]:
        best_p, cv_rmse, fold_rmses = grid_search_pca(
            model_name, model_cls, grid, folds, feat_all, n_comp_all,
        )
        best_p.update(extra)
        res = train_final_pca(
            model_name, model_cls, best_p, train_val_df, test_df,
            feat_all, n_comp_all, output_dir, "pca_all",
        )
        res["cv_mean_rmse"] = cv_rmse
        res["cv_fold_rmses"] = fold_rmses
        all_results.append(res)

    # =======================================================================
    # Experiment B: PCA WITHOUT life_exp — force multi-indicator learning
    # =======================================================================
    print("\n>>> Experiment B: PCA without life_exp (30 features)")
    feat_no_le = get_feature_cols(folds[0][0], drop_life_exp=True)
    print(f"  Features: {len(feat_no_le)}")

    pca_info_no_le = analyse_pca_variance(
        train_val_df[feat_no_le].values, feat_no_le, output_dir, "no_life_exp"
    )
    n_comp_no_le = pca_info_no_le["n_components_95pct"]
    print(f"  PCA 95% variance → {n_comp_no_le} components (from {len(feat_no_le)})")

    for model_name, model_cls, grid, extra in [
        ("Random Forest", RandomForestRegressor, RF_GRID, {"random_state": 42, "n_jobs": -1}),
        ("XGBoost", XGBRegressor, XGB_GRID, {"objective": "reg:squarederror", "random_state": 42, "verbosity": 0}),
    ]:
        best_p, cv_rmse, fold_rmses = grid_search_pca(
            model_name, model_cls, grid, folds, feat_no_le, n_comp_no_le,
        )
        best_p.update(extra)
        res = train_final_pca(
            model_name, model_cls, best_p, train_val_df, test_df,
            feat_no_le, n_comp_no_le, output_dir, "pca_no_life_exp",
        )
        res["cv_mean_rmse"] = cv_rmse
        res["cv_fold_rmses"] = fold_rmses
        all_results.append(res)

    # =======================================================================
    # Comparison
    # =======================================================================
    plot_all_comparison(all_results, output_dir)

    summary_path = output_dir / "pca_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"pca_variance_all": pca_info_all, "pca_variance_no_life_exp": pca_info_no_le, "results": all_results},
            f, indent=2,
        )

    print("\n" + "=" * 60)
    print("PCA Experiment Summary")
    print("=" * 60)
    for r in all_results:
        m = r["test_metrics"]
        print(
            f"  {r['model']:<15} [{r['tag']:<18}]  "
            f"PCs={r['n_components_actual']}  "
            f"RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
            f"R²={m['R2']:.4f}  MAPE={m['MAPE(%)']:.2f}%"
        )
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
