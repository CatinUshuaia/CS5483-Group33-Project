from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data_preprocessing" / "dataset" / "processeddataset"
OUTPUT_DIR = BASE_DIR / "modeling" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_xy(df, target_col="life_exp_next_year"):
    drop_cols = [target_col]
    if "year" in df.columns:
        drop_cols.append("year")
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


def main():
    train_df_1 = pd.read_csv(
        DATA_DIR / "wdi_china_lifeexp_model_ready_fold4_train_clip.csv"
    )
    train_df_2 = pd.read_csv(
        DATA_DIR / "wdi_china_lifeexp_model_ready_fold4_val_clip.csv"
    )
    test_df = pd.read_csv(
        DATA_DIR / "wdi_china_lifeexp_model_ready_test_clip.csv"
    )

    final_train_df = pd.concat([train_df_1, train_df_2], axis=0, ignore_index=True)

    X_train, y_train = load_xy(final_train_df)
    X_test, y_test = load_xy(test_df)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = rmse(y_test, test_pred)

    print("=== Final RF Test Result (clip) ===")
    print(f"Test MAE : {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

    pred_df = pd.DataFrame({
        "year": test_df["year"].values if "year" in test_df.columns else np.arange(len(test_df)),
        "y_true": y_test.values,
        "y_pred": test_pred
    })
    print("\n=== Test Predictions ===")
    print(pred_df)

    pred_df.to_csv(OUTPUT_DIR / "rf_test_predictions_clip.csv", index=False)

    feature_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n=== Top Feature Importances ===")
    print(feature_importance_df.head(10))

    feature_importance_df.to_csv(
        OUTPUT_DIR / "rf_feature_importance_clip.csv",
        index=False
    )


if __name__ == "__main__":
    main()