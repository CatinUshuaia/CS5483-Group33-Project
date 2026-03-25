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
│   │   ├── wdi_china_lifeexp_model_ready_no_clip.csv
│   │   └── processeddataset/
│   ├── preprocessing_scripts/
│   │   ├── preprocess_china_lifeexp.py
│   │   └── split_time_series_datasets.py
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

## TimeSeriesSplit Dataset Export

Export fold-level train/validation files with `TimeSeriesSplit`, then perform clip/scale inside each fold (fit on fold train only):

```bash
python data_preprocessing/preprocessing_scripts/split_time_series_datasets.py --test-start-year 2019 --n-splits 4
```

Default behavior:

- Input base dataset:
  - `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv`
- Train/validation period: years before `2019`
- Final test period: `2019-2022`
- Output folder: `data_preprocessing/dataset/processeddataset/`

## License

This project is licensed under the MIT License. See `LICENSE`.
