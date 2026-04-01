# Baseline Models Experiment Report

> **Date**: 2026-04-01
> **Task**: Predict China next-year life expectancy (`life_exp_next_year`)
> **Models**: Linear Regression (OLS), Ridge, Lasso
> **Data variant**: `clip_scaled` (IQR 裁剪 + StandardScaler 标准化，适合线性模型)
> **CV strategy**: 4-fold TimeSeriesSplit (pre-built folds, train period 1995–2018)
> **Test set**: 2019–2022 (4 samples)

---

## 1. Experimental Setup

| Item | Detail |
|------|--------|
| Features | 31 (life_exp + 10 base WDI indicators + 10 lag1 + 10 pct_change) |
| Fold sizes (train, val) | Fold 1: (8, 4), Fold 2: (12, 4), Fold 3: (16, 4), Fold 4: (20, 4) |
| Full train+val | 24 rows (1995–2018) |
| Test | 4 rows (2019–2022) |
| Data variant | `clip_scaled`：先 IQR 裁剪异常值，再 StandardScaler 标准化（fit on train only） |

### 为什么使用 clip_scaled 变体？

- 线性模型对特征尺度敏感，标准化后系数可直接比较重要性
- IQR 裁剪减少异常值对 OLS 的杠杆效应
- Ridge/Lasso 的正则化项 α‖w‖ 在标准化后更公平地约束每个特征

### Alpha 搜索空间（Ridge / Lasso）

```
[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
```

---

## 2. Cross-Validation Results

| Model | Best Alpha | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean RMSE** |
|-------|-----------|--------|--------|--------|--------|--------------|
| Linear Regression | — | 0.5763 | 1.5756 | 0.9931 | 0.3970 | **0.8855** |
| Ridge | 0.5 | 0.7787 | 1.6077 | 0.1717 | 0.2893 | **0.7119** |
| Lasso | 0.05 | 0.1853 | 0.1359 | 0.3164 | 0.1180 | **0.1889** |

**关键发现**：
- Lasso 的 CV RMSE（0.189）大幅领先，约为 Linear 的 1/5、Ridge 的 1/4
- Linear Regression 的 CV 方差最大（Fold 2 高达 1.58），说明无正则化时过拟合严重
- Ridge 在 Fold 3/4 表现好（< 0.3），但在 Fold 2 也偏高

---

## 3. Test Set Metrics (2019–2022)

| Metric | Linear Regression | Ridge | **Lasso** |
|--------|------------------|-------|-----------|
| **RMSE** | 1.5016 | 2.0398 | **0.7993** |
| **MAE** | 1.1845 | 1.8475 | **0.6951** |
| **R²** | -250.30 | -462.68 | **-70.21** |
| **MAPE (%)** | 1.52 | 2.37 | **0.89** |

### Per-Year Predictions

**Linear Regression**:

| Year | Actual | Predicted | Residual |
|------|--------|-----------|----------|
| 2019 | 78.019 | 77.411 | +0.608 |
| 2020 | 78.117 | 78.113 | +0.004 |
| 2021 | 78.202 | 75.877 | +2.325 |
| 2022 | 77.953 | 76.152 | +1.801 |

**Ridge** (α=0.5):

| Year | Actual | Predicted | Residual |
|------|--------|-----------|----------|
| 2019 | 78.019 | 76.490 | +1.529 |
| 2020 | 78.117 | 77.496 | +0.621 |
| 2021 | 78.202 | 75.274 | +2.928 |
| 2022 | 77.953 | 75.641 | +2.312 |

**Lasso** (α=0.05):

| Year | Actual | Predicted | Residual |
|------|--------|-----------|----------|
| 2019 | 78.019 | 77.033 | +0.986 |
| 2020 | 78.117 | 77.780 | +0.337 |
| 2021 | 78.202 | 77.022 | +1.180 |
| 2022 | 77.953 | 77.676 | +0.277 |

---

## 4. Analysis

### 4.1 Lasso 显著优于其他两个基线模型

Lasso 在所有指标上大幅领先：RMSE 0.80（Linear 1.50, Ridge 2.04），MAPE 0.89%。这得益于 L1 正则化的**特征选择能力**——Lasso 自动将 31 个特征中的 22 个系数压缩为零，只保留了 **9 个非零特征**，有效避免了小样本下的过拟合。

### 4.2 Linear Regression 的过拟合问题

OLS 没有正则化约束，在 24 样本 31 特征的场景下拟合了噪声。CV 方差大（Fold 2 RMSE=1.58 vs Fold 4=0.40），且在 2021/2022 年预测偏差超过 2 岁。系数绝对值大且符号不稳定（如 `urban_pop_pct` 系数为 -8.78），反映了多重共线性导致的估计不稳定。

### 4.3 Ridge 表现反而最差

Ridge（L2 正则化）虽然缩小了系数，但保留了所有 31 个特征，无法做特征选择。在高度共线的特征空间中（每个指标有原值/lag1/pct_change 三个版本），Ridge 将权重均匀分散到共线特征上，测试误差反而增大。

### 4.4 R² 为负的原因

同高级模型实验一致，R² 负值是由测试集极小样本量（4 个点，方差极小）导致的统计假象。MAPE 和 MAE 是此场景下更可靠的指标。

### 4.5 Lasso 的特征选择结果

Lasso（α=0.05）自动选出 9 个非零特征（共 31 个），实现了隐式的特征筛选，这是 Lasso 在小样本高维场景下的核心优势。具体哪些特征被保留可查看 `lasso/results.json` 中的 `coefficients` 字段。

### 4.6 所有模型的系统性低估

三个模型对所有测试年份均**正残差**（低估），说明线性模型在外推时倾向保守。这可能因为 2019-2022 的寿命增长趋势与训练期不完全一致。

---

## 5. 与高级模型的初步对比

| Model | Type | RMSE | MAE | MAPE (%) |
|-------|------|------|-----|---------|
| **Lasso (α=0.05)** | Baseline | **0.7993** | **0.6951** | **0.89** |
| Linear Regression | Baseline | 1.5016 | 1.1845 | 1.52 |
| Ridge (α=0.5) | Baseline | 2.0398 | 1.8475 | 2.37 |

Lasso 作为基线模型的 RMSE（0.80）已经相当有竞争力，有待与成员 3 的 RF/XGBoost 结果做统一对比。

---

## 6. Output Artifacts

| File | Description |
|------|-------------|
| `summary.json` | 三个模型的完整结果（JSON） |
| `model_comparison.png` | 三模型指标对比柱状图 |
| `alpha_tuning_curves.png` | Ridge/Lasso 的 α 调优曲线 |
| `coefficient_heatmap.png` | 三模型系数热力图对比 |
| `linear_regression/` | OLS 的预测、系数图、残差图 |
| `ridge/` | Ridge 的预测、系数图、残差图 |
| `lasso/` | Lasso 的预测、系数图、残差图 |
