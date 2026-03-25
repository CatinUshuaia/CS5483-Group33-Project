import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


INDICATORS: Dict[str, str] = {
    "life_exp": "SP.DYN.LE00.IN",  # target (current year)
    "gdp_per_capita": "NY.GDP.PCAP.KD",
    "health_exp_gdp": "SH.XPD.CHEX.GD.ZS",
    "health_exp_percap": "SH.XPD.CHEX.PC.CD",
    "pm25": "EN.ATM.PM25.MC.M3",
    "urban_pop_pct": "SP.URB.TOTL.IN.ZS",
    "pop_65_plus_pct": "SP.POP.65UP.TO.ZS",
    "unemployment": "SL.UEM.TOTL.ZS",
    "med_beds": "SH.MED.BEDS.ZS",
    "infant_mortality": "SP.DYN.IMRT.IN",
    "total_fertility": "SP.DYN.TFRT.IN",
}

BASE_FEATURES: List[str] = [
    "gdp_per_capita",
    "health_exp_gdp",
    "health_exp_percap",
    "pm25",
    "urban_pop_pct",
    "pop_65_plus_pct",
    "unemployment",
    "med_beds",
    "infant_mortality",
    "total_fertility",
]

# Features that should never be negative.
NON_NEGATIVE_FEATURES: List[str] = BASE_FEATURES + ["life_exp"]

# Features represented as percentages (expected in [0, 100]).
PERCENT_FEATURES: List[str] = [
    "health_exp_gdp",
    "urban_pop_pct",
    "pop_65_plus_pct",
    "unemployment",
]


def fetch_wdi_indicator(country: str, indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch one WDI indicator from World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator_code}"
    params = {
        "format": "json",
        "per_page": 20000,
        "date": f"{start_year}:{end_year}",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Unexpected API response for {indicator_code}")

    rows = []
    for rec in data[1]:
        year_raw = rec.get("date")
        if year_raw is None:
            continue
        try:
            year = int(year_raw)
        except ValueError:
            continue
        rows.append({"year": year, "value": rec.get("value")})

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def latest_non_null_year(df: pd.DataFrame, value_col: str) -> int | None:
    """Return latest year where value is non-null."""
    valid = df[df[value_col].notna()]
    if valid.empty:
        return None
    return int(valid["year"].max())


def first_round_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Impute missing values with forward fill then median."""
    df = df.copy()
    df[cols] = df[cols].ffill()
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def apply_data_quality_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Fix simple data issues: duplicates, impossible values, and outliers."""
    out = df.copy()
    quality_stats: Dict[str, int] = {
        "duplicate_year_rows_removed": 0,
        "negative_values_fixed_to_nan": 0,
        "percent_out_of_range_fixed_to_nan": 0,
        "outlier_values_clipped_iqr": 0,
    }

    # 1) Remove duplicate year rows.
    before = len(out)
    out = out.drop_duplicates(subset=["year"], keep="first").reset_index(drop=True)
    quality_stats["duplicate_year_rows_removed"] = before - len(out)

    # 2) Non-negative rule: negatives are treated as invalid and set to NaN.
    for col in NON_NEGATIVE_FEATURES:
        if col not in out.columns:
            continue
        mask = out[col].notna() & (out[col] < 0)
        count = int(mask.sum())
        if count > 0:
            out.loc[mask, col] = pd.NA
            quality_stats["negative_values_fixed_to_nan"] += count

    # 3) Percentage rule: values outside [0, 100] are treated as invalid.
    for col in PERCENT_FEATURES:
        if col not in out.columns:
            continue
        mask = out[col].notna() & ((out[col] < 0) | (out[col] > 100))
        count = int(mask.sum())
        if count > 0:
            out.loc[mask, col] = pd.NA
            quality_stats["percent_out_of_range_fixed_to_nan"] += count

    return out, quality_stats


def clip_outliers_iqr(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, int]:
    """Clip extreme values by IQR rule."""
    out = df.copy()
    clipped = 0
    for col in cols:
        if col not in out.columns:
            continue
        s = out[col]
        if s.dropna().empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = s.notna() & ((s < lower) | (s > upper))
        count = int(mask.sum())
        if count > 0:
            out[col] = s.clip(lower=lower, upper=upper)
            clipped += count
    return out, clipped


def replace_section(text: str, start_marker: str, end_marker: str, new_block: str) -> str:
    """Replace text between two markers (inclusive block content only)."""
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return text
    content_start = start_idx + len(start_marker)
    return text[:content_start] + "\n" + new_block + "\n" + text[end_idx:]


def update_preprocessing_details(
    details_path: Path,
    rows_total: int,
    year_min: int,
    year_max: int,
    feature_count: int,
    quality_stats: Dict[str, int],
    missing_raw_base: int,
    latest_year_map: Dict[str, int | None],
) -> None:
    """Dynamically update quality and metadata sections in PREPROCESSING_DETAILS.md."""
    if not details_path.exists():
        return

    text = details_path.read_text(encoding="utf-8")

    quality_block = "\n".join(
        [
            f"- `rows_total = {rows_total}`",
            f"- `duplicate_year_rows_removed = {quality_stats['duplicate_year_rows_removed']}`",
            f"- `negative_values_fixed_to_nan = {quality_stats['negative_values_fixed_to_nan']}`",
            f"- `percent_out_of_range_fixed_to_nan = {quality_stats['percent_out_of_range_fixed_to_nan']}`",
            f"- `outlier_values_detected_by_iqr = {quality_stats['outlier_values_clipped_iqr']}`",
            f"- `missing_values_final_raw_no_clip = {missing_raw_base}`",
        ]
    )

    summary_block = "\n".join(
        [
            f"- 年份范围（建模样本）：`{year_min}-{year_max}`",
            f"- 特征列数量（不含 `year` 与标签）：{feature_count}",
            "- 目标列：`life_exp_next_year`",
            "- 交付基础数据：未标准化 + 未裁剪（fold内再做 clip/scale）",
        ]
    )

    ordered_indicators = [
        "life_exp",
        "gdp_per_capita",
        "health_exp_gdp",
        "health_exp_percap",
        "pm25",
        "urban_pop_pct",
        "pop_65_plus_pct",
        "unemployment",
        "med_beds",
        "infant_mortality",
        "total_fertility",
    ]
    latest_block = "\n".join(
        [f"- `{name}`: {latest_year_map.get(name)}" for name in ordered_indicators]
    )

    text = replace_section(text, "<!-- AUTO_QUALITY_START -->", "<!-- AUTO_QUALITY_END -->", quality_block)
    text = replace_section(text, "<!-- AUTO_SUMMARY_START -->", "<!-- AUTO_SUMMARY_END -->", summary_block)
    text = replace_section(text, "<!-- AUTO_LATEST_YEAR_START -->", "<!-- AUTO_LATEST_YEAR_END -->", latest_block)

    details_path.write_text(text, encoding="utf-8")


def build_dataset(
    country: str, start_year: int, end_year: int
) -> Tuple[pd.DataFrame, List[str], Dict[str, int | None], Dict[str, int]]:
    """Build base modeling dataset (raw + no-clip) for fold-level transforms."""
    frames = []
    latest_year_map: Dict[str, int | None] = {}
    for name, code in INDICATORS.items():
        tmp = fetch_wdi_indicator(country=country, indicator_code=code, start_year=start_year, end_year=end_year)
        latest_year_map[name] = latest_non_null_year(tmp, "value")
        tmp = tmp.rename(columns={"value": name})
        frames.append(tmp)

    merged = frames[0]
    for i in range(1, len(frames)):
        merged = merged.merge(frames[i], on="year", how="outer")

    merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)].sort_values("year").reset_index(drop=True)

    # Step 0: basic data quality rules
    merged, quality_stats = apply_data_quality_rules(merged)

    # Step 1: impute base indicators first
    numeric_base = ["life_exp"] + BASE_FEATURES
    merged = first_round_impute(merged, numeric_base)

    # Step 2: label y_{t+1}
    merged["life_exp_next_year"] = merged["life_exp"].shift(-1)

    # Step 3: lag features x_{t-1}
    for col in BASE_FEATURES:
        merged[f"{col}_lag1"] = merged[col].shift(1)

    # Step 4: pct_change features
    for col in BASE_FEATURES:
        merged[f"{col}_pct_change"] = merged[col].pct_change()

    # Step 5: remove last year (no next-year label)
    model_df = merged.dropna(subset=["life_exp_next_year"]).copy()

    # Step 6: second round impute for engineered features
    feature_cols = [c for c in model_df.columns if c not in ["year", "life_exp_next_year"]]
    model_df = first_round_impute(model_df, feature_cols)

    # Detect potential outliers for reporting; do not clip here.
    _, clipped_count = clip_outliers_iqr(model_df, feature_cols)
    quality_stats["outlier_values_clipped_iqr"] = clipped_count

    return model_df, feature_cols, latest_year_map, quality_stats


def main() -> None:
    script_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Preprocess WDI data for China next-year life expectancy prediction.")
    parser.add_argument("--country", type=str, default="CHN", help="Country code, default: CHN")
    parser.add_argument("--start-year", type=int, default=1995, help="Start year, default: 1995")
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="End year, default: 2023 (recommended for March 2026 data completeness)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory, default: data_preprocessing/dataset",
    )
    parser.add_argument(
        "--details-path",
        type=str,
        default=None,
        help="Path to PREPROCESSING_DETAILS.md, default: data_preprocessing/PREPROCESSING_DETAILS.md",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else (script_root / "dataset")
    outdir.mkdir(parents=True, exist_ok=True)

    raw_df_base, feature_cols, latest_year_map, quality_stats = build_dataset(
        country=args.country,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    base_path = outdir / "wdi_china_lifeexp_model_ready_no_clip.csv"

    raw_df_base.to_csv(base_path, index=False)

    print("Preprocessing done.")
    print(f"- Base dataset (raw no-clip): {base_path}")
    print(f"- Rows: {len(raw_df_base)}")
    print(f"- Year range: {int(raw_df_base['year'].min())} - {int(raw_df_base['year'].max())}")
    print(f"- Feature count: {len(feature_cols)}")
    print(f"- Quality duplicate_year_rows_removed: {quality_stats['duplicate_year_rows_removed']}")
    print(f"- Quality negative_values_fixed_to_nan: {quality_stats['negative_values_fixed_to_nan']}")
    print(f"- Quality percent_out_of_range_fixed_to_nan: {quality_stats['percent_out_of_range_fixed_to_nan']}")
    print(f"- Quality outlier_values_detected_by_iqr: {quality_stats['outlier_values_clipped_iqr']}")
    print(f"- Missing values (raw no-clip): {int(raw_df_base.isna().sum().sum())}")

    # Keep documentation metrics synchronized after each run.
    details_path = Path(args.details_path) if args.details_path else (script_root / "PREPROCESSING_DETAILS.md")
    update_preprocessing_details(
        details_path=details_path,
        rows_total=len(raw_df_base),
        year_min=int(raw_df_base["year"].min()),
        year_max=int(raw_df_base["year"].max()),
        feature_count=len(feature_cols),
        quality_stats=quality_stats,
        missing_raw_base=int(raw_df_base.isna().sum().sum()),
        latest_year_map=latest_year_map,
    )

    print("- Latest non-null year by indicator:")
    for name, year in latest_year_map.items():
        print(f"  - {name}: {year}")


if __name__ == "__main__":
    main()
