# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project follows semantic-style version sections.

## [Unreleased]

### Changed
- Switched to the stricter pipeline: keep one base dataset (`raw + no-clip`) and move clip/scale into fold-level processing.
- Updated `preprocess_china_lifeexp.py` to export only `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv`.
- Reworked `split_time_series_datasets.py` to apply IQR clipping and StandardScaler inside each `TimeSeriesSplit` fold (fit on train fold only).
- Exported fold outputs and final test variants (`no_clip`, `clip`, `no_clip_scaled`, `clip_scaled`) under `data_preprocessing/dataset/processeddataset/`.
- Removed redundant full-sample standardized artifacts from previous steps.

## [0.1.0] - 2026-03-25

### Added
- Initial repository setup and first commit.
- Data preprocessing pipeline script for WDI-based life expectancy task.
- Generated four model-ready datasets (clip/no-clip, raw/scaled).
- Documentation files for preprocessing details and literature-to-feature mapping.

### Changed
- Reorganized folder structure to standardized naming:
  - `data_preprocessing/`
  - `data_preprocessing/dataset/`
  - `data_preprocessing/preprocessing_scripts/`
  - `data_preprocessing/feature_selection_literature/`
- Moved `FEATURE_SELECTION_LITERATURE_NOTE.md` to `data_preprocessing/`.
- Updated script/documentation paths to match the new structure.
