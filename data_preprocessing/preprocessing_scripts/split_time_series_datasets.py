import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def canonical_prefix_from_input(input_csv: Path) -> str:
    """Normalize file prefix to avoid contradictory naming like no_clip + clip."""
    stem = input_csv.stem
    if stem.endswith("_no_clip"):
        return stem[: -len("_no_clip")]
    return stem


def compute_clip_bounds(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in feature_cols:
        s = train_df[col]
        if s.dropna().empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        bounds[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    return bounds


def apply_clip_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def apply_scaler(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_out = train_df.copy()
    other_out = other_df.copy()
    train_out[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    other_out[feature_cols] = scaler.transform(other_df[feature_cols])
    return train_out, other_out


def export_fold_outputs(
    stem: str,
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    clip_bounds = compute_clip_bounds(train_df, feature_cols)
    train_clip = apply_clip_bounds(train_df, clip_bounds)
    val_clip = apply_clip_bounds(val_df, clip_bounds)

    train_no_clip_scaled, val_no_clip_scaled = apply_scaler(train_df, val_df, feature_cols)
    train_clip_scaled, val_clip_scaled = apply_scaler(train_clip, val_clip, feature_cols)

    train_df.to_csv(output_dir / f"{stem}_fold{fold_idx}_train_no_clip.csv", index=False)
    val_df.to_csv(output_dir / f"{stem}_fold{fold_idx}_val_no_clip.csv", index=False)
    train_clip.to_csv(output_dir / f"{stem}_fold{fold_idx}_train_clip.csv", index=False)
    val_clip.to_csv(output_dir / f"{stem}_fold{fold_idx}_val_clip.csv", index=False)
    train_no_clip_scaled.to_csv(output_dir / f"{stem}_fold{fold_idx}_train_no_clip_scaled.csv", index=False)
    val_no_clip_scaled.to_csv(output_dir / f"{stem}_fold{fold_idx}_val_no_clip_scaled.csv", index=False)
    train_clip_scaled.to_csv(output_dir / f"{stem}_fold{fold_idx}_train_clip_scaled.csv", index=False)
    val_clip_scaled.to_csv(output_dir / f"{stem}_fold{fold_idx}_val_clip_scaled.csv", index=False)


def export_final_test_outputs(
    stem: str,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    clip_bounds = compute_clip_bounds(train_val_df, feature_cols)
    train_val_clip = apply_clip_bounds(train_val_df, clip_bounds)
    test_clip = apply_clip_bounds(test_df, clip_bounds)

    _, test_no_clip_scaled = apply_scaler(train_val_df, test_df, feature_cols)
    _, test_clip_scaled = apply_scaler(train_val_clip, test_clip, feature_cols)

    test_df.to_csv(output_dir / f"{stem}_test_no_clip.csv", index=False)
    test_clip.to_csv(output_dir / f"{stem}_test_clip.csv", index=False)
    test_no_clip_scaled.to_csv(output_dir / f"{stem}_test_no_clip_scaled.csv", index=False)
    test_clip_scaled.to_csv(output_dir / f"{stem}_test_clip_scaled.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TimeSeriesSplit folds and fold-level clip/scale outputs from base raw dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv",
        help="Base dataset path (raw + no-clip).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_preprocessing/dataset/processeddataset",
        help="Output directory for split CSV files.",
    )
    parser.add_argument(
        "--test-start-year",
        type=int,
        default=2019,
        help="First year included in final test set.",
    )
    parser.add_argument("--n-splits", type=int, default=4, help="Number of TimeSeriesSplit folds.")
    args = parser.parse_args()

    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Base dataset not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "year" not in df.columns or "life_exp_next_year" not in df.columns:
        raise ValueError("Input dataset must include `year` and `life_exp_next_year` columns.")

    df = df.sort_values("year").reset_index(drop=True)
    train_val_df = df[df["year"] < args.test_start_year].copy()
    test_df = df[df["year"] >= args.test_start_year].copy()
    if train_val_df.empty:
        raise ValueError(f"Input dataset has empty train/validation period before {args.test_start_year}.")
    if test_df.empty:
        raise ValueError(f"Input dataset has empty test period from {args.test_start_year}.")
    if len(train_val_df) <= args.n_splits:
        raise ValueError(
            f"Input dataset has {len(train_val_df)} train/validation rows; "
            f"not enough for n_splits={args.n_splits}."
        )

    feature_cols = [c for c in df.columns if c not in ["year", "life_exp_next_year"]]
    stem = canonical_prefix_from_input(input_csv)

    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    fold_count = 0
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(train_val_df), start=1):
        fold_count += 1
        fold_train_df = train_val_df.iloc[train_idx].copy()
        fold_val_df = train_val_df.iloc[val_idx].copy()
        export_fold_outputs(
            stem=stem,
            fold_idx=fold_idx,
            train_df=fold_train_df,
            val_df=fold_val_df,
            feature_cols=feature_cols,
            output_dir=output_dir,
        )

    export_final_test_outputs(
        stem=stem,
        train_val_df=train_val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        output_dir=output_dir,
    )

    print(f"[DONE] {input_csv.name}")
    print(f"  folds={fold_count} (TimeSeriesSplit)")
    print(f"  train/val years={int(train_val_df['year'].min())}-{int(train_val_df['year'].max())}")
    print(f"  test years={int(test_df['year'].min())}-{int(test_df['year'].max())}")
    print(f"\nAll TimeSeriesSplit files saved to: {output_dir}")


if __name__ == "__main__":
    main()
