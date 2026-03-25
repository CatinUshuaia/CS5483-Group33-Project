# 中国下一年预期寿命预测：数据预处理说明

## 0. 快速运行
- 在 `CS5483-Group33-Project` 目录执行依赖安装：`pip install pandas requests scikit-learn`
- 执行脚本（默认输出到 `data_preprocessing/dataset`）：`python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py`
- 指定参数执行：`python data_preprocessing/preprocessing_scripts/preprocess_china_lifeexp.py --country CHN --start-year 1995 --end-year 2023 --outdir data_preprocessing/dataset --details-path data_preprocessing/PREPROCESSING_DETAILS.md`

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
- 对建模特征执行 IQR 裁剪（clip 版本）
- 同时保留不做异常值裁剪的 no-clip 版本

### 3.7 标准化
- 对特征列执行 Z-score 标准化
- clip 与 no-clip 各自生成一份标准化数据

## 4. 交付文件

### 4.1 最终数据集（裁剪版）
- `data_preprocessing/dataset/wdi_china_lifeexp_model_ready.csv`（未标准化，异常值裁剪版）
- `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_scaled.csv`（标准化，异常值裁剪版）

### 4.2 最终数据集（未裁剪版）
- `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_no_clip.csv`（未标准化，未裁剪版）
- `data_preprocessing/dataset/wdi_china_lifeexp_model_ready_scaled_no_clip.csv`（标准化，未裁剪版）

## 5. 质量报告整合结果（当前版本）
<!-- AUTO_QUALITY_START -->
- `rows_total = 28`
- `duplicate_year_rows_removed = 0`
- `negative_values_fixed_to_nan = 0`
- `percent_out_of_range_fixed_to_nan = 0`
- `outlier_values_clipped_iqr = 47`
- `missing_values_final_raw_clip = 0`
- `missing_values_final_scaled_clip = 0`
- `missing_values_final_raw_no_clip = 0`
- `missing_values_final_scaled_no_clip = 0`
<!-- AUTO_QUALITY_END -->

## 6. 当前可用数据说明
<!-- AUTO_SUMMARY_START -->
- 年份范围（建模样本）：`1995-2022`
- 特征列数量（不含 `year` 与标签）：31
- 目标列：`life_exp_next_year`
- clip/no-clip 两套数据均可直接交付建模
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
- 标准化当前在全样本上执行，后续若做严格时间序列评估，需在训练集 `fit` 后再应用到验证/测试。
- `pm25` 最新非空年份是 2020，2021-2023 为补值延续，需在结果解释中注明。
- IQR 裁剪共改动 47 个特征值，建模时建议同时对比 clip/no-clip 两套数据结果。
