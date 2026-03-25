# 中国下一年预期寿命预测：数据预处理说明

## 0. 快速运行
- 在 `CS5483-Group33-Project` 目录执行依赖安装：`pip install pandas requests scikit-learn`
- 执行脚本（默认输出到 `data_preprocessing/dataset`）：`python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py`
- 指定参数执行：`python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py --country CHN --start-year 1995 --end-year 2023 --outdir data_preprocessing/dataset --details-path data_preprocessing/PREPROCESSING_DETAILS.md`
- 导出 TimeSeriesSplit 划分：`python data_preprocessing/preprocessing_scripts/split_time_series_datasets.py --test-start-year 2019 --n-splits 4`

## 0.1 划分参数规范（固定）
- 本项目统一固定：`test_start_year = 2019`
- 即：`1995-2018` 用于 `TimeSeriesSplit` 的 train/validation，`2019-2022` 作为最终测试集
- 除非全组讨论并统一更新文档，否则不要更改该参数，以保证实验可比性

## 1. 任务与数据范围
- 任务类型：回归
- 目标变量：`life_exp_next_year`
- 标签构造：`life_exp_next_year = life_exp.shift(-1)`
- 国家：`CHN`
- 时间范围：`1995-2023`
- 数据源：World Bank WDI（API 拉取）

## 2. 使用的基础特征（10个）
- `NY.GDP.PCAP.KD`（人均 GDP）
- `SH.XPD.CHEX.GD.ZS`（卫生支出占 GDP）
- `SH.XPD.CHEX.PC.CD`（人均卫生支出）
- `EN.ATM.PM25.MC.M3`（PM2.5 暴露）
- `SP.URB.TOTL.IN.ZS`（城市人口占比）
- `SP.POP.65UP.TO.ZS`（65 岁以上人口占比）
- `SL.UEM.TOTL.ZS`（失业率）
- `SH.MED.BEDS.ZS`（每千人病床数）
- `SP.DYN.IMRT.IN`（婴儿死亡率）
- `SP.DYN.TFRT.IN`（总和生育率）

## 3. 数据预处理已执行步骤

### 3.1 数据拉取与集成
- 对每个指标单独调用 WDI API 拉取中国年度数据
- 保留 `year` + 指标值
- 按 `year` 外连接合并为单一宽表

### 3.2 数据质量规则处理
- 按 `year` 去重
- 非负约束指标（如 GDP、病床数、婴儿死亡率等）若出现负值，置为缺失
- 百分比指标（如城市化率、失业率等）若不在 `[0,100]`，置为缺失

### 3.3 缺失值处理
- 对 `life_exp + 10 个基础特征` 执行：前向填充 -> 中位数填充
- 构造工程特征后再次执行：前向填充 -> 中位数填充

### 3.4 标签与样本处理
- 构造 `life_exp_next_year = life_exp.shift(-1)`
- 删除最后一年（无 next-year 标签）

### 3.5 特征工程
- 滞后特征：为 10 个基础特征生成 `*_lag1`
- 变化率特征：为 10 个基础特征生成 `*_pct_change`

### 3.6 异常值处理
- 基础预处理阶段不直接裁剪，保留原始 no-clip 数据
- 在 TimeSeriesSplit 每个 fold 内，仅用训练折拟合 IQR 边界并应用到验证/测试

### 3.7 标准化
- 基础预处理阶段不直接标准化
- 在 TimeSeriesSplit 每个 fold 内，仅用训练折拟合 StandardScaler 并应用到验证/测试

## 4. 交付文件

### 4.1 基础数据集（未标准化，未裁剪）
- `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv`

### 4.2 TimeSeriesSplit 导出文件（fold 内动态 clip/scale）
- 输出目录：`data_preprocessing/dataset/processeddataset/`
- 生成 `fold1..foldN` 的 train/val 文件，包含 `no_clip`、`clip`、`no_clip_scaled`、`clip_scaled` 四类
- 生成最终测试文件：`*_test_no_clip.csv`、`*_test_clip.csv`、`*_test_no_clip_scaled.csv`、`*_test_clip_scaled.csv`

### 4.3 划分元数据清单（n_splits=4, test_start_year=2019）
- 基础样本年份：`1995-2022`（28 行）
- Train/Validation 总区间：`1995-2018`（24 行）
- Final Test 区间：`2019-2022`（4 行）

| Fold | Train 年份 | Train 行数 | Val 年份 | Val 行数 |
|---|---|---:|---|---:|
| 1 | 1995-2002 | 8 | 2003-2006 | 4 |
| 2 | 1995-2006 | 12 | 2007-2010 | 4 |
| 3 | 1995-2010 | 16 | 2011-2014 | 4 |
| 4 | 1995-2014 | 20 | 2015-2018 | 4 |

## 5. 质量报告整合结果（当前版本）
<!-- AUTO_QUALITY_START -->
- `rows_total = 28`
- `duplicate_year_rows_removed = 0`
- `negative_values_fixed_to_nan = 0`
- `percent_out_of_range_fixed_to_nan = 0`
- `outlier_values_detected_by_iqr = 47`
- `missing_values_final_raw_no_clip = 0`
<!-- AUTO_QUALITY_END -->

## 6. 当前可用数据说明
<!-- AUTO_SUMMARY_START -->
- 年份范围（建模样本）：`1995-2022`
- 特征列数量（不含 `year` 与标签）：31
- 目标列：`life_exp_next_year`
- 交付基础数据：未标准化 + 未裁剪（fold内再做 clip/scale）
<!-- AUTO_SUMMARY_END -->

## 7. 指标最新非空年份记录
<!-- AUTO_LATEST_YEAR_START -->
- `life_exp`: 2023
- `gdp_per_capita`: 2023
- `health_exp_gdp`: 2023
- `health_exp_percap`: 2023
- `pm25`: 2020
- `urban_pop_pct`: 2023
- `pop_65_plus_pct`: 2023
- `unemployment`: 2023
- `med_beds`: 2023
- `infant_mortality`: 2023
- `total_fertility`: 2023
<!-- AUTO_LATEST_YEAR_END -->

## 8. 后续建模风险提示
- 当前流程已在 TimeSeriesSplit 内按训练折拟合并应用标准化/裁剪，后续新增模型时应保持同样规则以避免时间泄漏。
- `pm25` 最新非空年份是 2020，2021-2023 为补值延续，需在结果解释中注明。
- IQR 异常值检测统计值为 47（用于参考），建模时建议对比 clip/no-clip 两套结果的稳健性。
