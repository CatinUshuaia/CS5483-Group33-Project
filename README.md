# CS5483 Group33 Project

China next-year life expectancy prediction project with data preprocessing pipeline based on World Bank WDI indicators.

## Project Overview

This repository currently focuses on the preprocessing stage for model-ready datasets.

- Data source: World Bank WDI API
- Target variable: `life_exp_next_year`
- Country: `CHN`
- Modeling sample years: `1995-2022` (label uses next year)

## Repository Structure

```text
CS5483-Group33-Project/
├── data_preprocessing/
│   ├── dataset/
│   │   ├── wdi_china_lifeexp_model_ready.csv
│   │   ├── wdi_china_lifeexp_model_ready_scaled.csv
│   │   ├── wdi_china_lifeexp_model_ready_no_clip.csv
│   │   └── wdi_china_lifeexp_model_ready_scaled_no_clip.csv
│   ├── preprocessing_scripts/
│   │   └── preprocess_china_lifeexp.py
│   ├── FEATURE_SELECTION_LITERATURE_NOTE.md
│   ├── PREPROCESSING_DETAILS.md
│   └── feature_selection_literature/
└── ...
```

## Environment Setup

Recommended Python version: 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Preprocessing

From repository root:

```bash
python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py
```

Custom run:

```bash
python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py --country CHN --start-year 1995 --end-year 2023 --outdir data_preprocessing/dataset --details-path data_preprocessing/PREPROCESSING_DETAILS.md
```

## Output Files

The script generates four datasets under `data_preprocessing/dataset/`:

- `wdi_china_lifeexp_model_ready.csv` (raw, clipped)
- `wdi_china_lifeexp_model_ready_scaled.csv` (scaled, clipped)
- `wdi_china_lifeexp_model_ready_no_clip.csv` (raw, no-clip)
- `wdi_china_lifeexp_model_ready_scaled_no_clip.csv` (scaled, no-clip)

## Collaboration Workflow (Suggested)

- Create feature branch: `feature/<topic>`
- Commit with clear messages
- Open Pull Request to `main`
- At least one teammate review before merge

## License

This project is licensed under the MIT License. See `LICENSE`.
