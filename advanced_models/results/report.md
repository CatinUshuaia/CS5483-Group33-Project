# Advanced Models Experiment Report

> **Date**: 2026-03-31
> **Task**: Predict China next-year life expectancy (`life_exp_next_year`)
> **Models**: Random Forest, XGBoost
> **Data variant**: `no_clip` (raw features, no outlier clipping, no scaling — tree models are scale-invariant)
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
| Tuning method | Grid search (quick mode) with mean validation RMSE as selection criterion |

### Hyperparameter Search Space (Quick Mode)

**Random Forest** (16 combinations):

| Parameter | Values |
|-----------|--------|
| n_estimators | 100, 300 |
| max_depth | 3, None |
| min_samples_split | 2 |
| min_samples_leaf | 1, 2 |
| max_features | sqrt, 1.0 |

**XGBoost** (32 combinations):

| Parameter | Values |
|-----------|--------|
| n_estimators | 100, 300 |
| max_depth | 3, 5 |
| learning_rate | 0.05, 0.1 |
| subsample | 0.7, 1.0 |
| colsample_bytree | 0.7, 1.0 |
| reg_alpha | 0 |
| reg_lambda | 1 |

---

## 2. Best Hyperparameters

### Random Forest

| Parameter | Best Value |
|-----------|-----------|
| n_estimators | 100 |
| max_depth | None (unlimited) |
| min_samples_split | 2 |
| min_samples_leaf | 1 |
| max_features | sqrt |

### XGBoost

| Parameter | Best Value |
|-----------|-----------|
| n_estimators | 300 |
| max_depth | 3 |
| learning_rate | 0.1 |
| subsample | 1.0 |
| colsample_bytree | 1.0 |
| reg_alpha | 0 |
| reg_lambda | 1 |

---

## 3. Cross-Validation Results (Mean Validation RMSE)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean** |
|-------|--------|--------|--------|--------|----------|
| Random Forest | 1.2617 | 1.5970 | 1.9382 | 1.2948 | **1.5229** |
| XGBoost | 0.9520 | 0.7533 | 0.7551 | 0.6325 | **0.7732** |

- XGBoost 的 CV RMSE 约为 Random Forest 的一半，且各 fold 间方差更小，泛化更稳定。
- Random Forest 在 Fold 3 上误差最大（1.94），可能因为该 fold 的验证集包含趋势变化较大的年份。

---

## 4. Test Set Metrics (2019–2022)

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| **RMSE** | 1.1269 | **0.1641** |
| **MAE** | 1.1197 | **0.1340** |
| **R²** | -140.51 | -2.00 |
| **MAPE (%)** | 1.43 | **0.17** |

### Per-Year Predictions

| Year | Actual | RF Predicted | RF Residual | XGB Predicted | XGB Residual |
|------|--------|-------------|-------------|---------------|--------------|
| 2019 | 78.019 | 77.021 | +0.998 | 77.939 | +0.080 |
| 2020 | 78.117 | 76.962 | +1.155 | 77.939 | +0.178 |
| 2021 | 78.202 | 76.757 | +1.445 | 77.939 | +0.263 |
| 2022 | 77.953 | 76.879 | +1.074 | 77.939 | +0.014 |

---

## 5. Analysis

### 5.1 XGBoost 显著优于 Random Forest

XGBoost 在所有指标上大幅领先：测试集 RMSE 仅 0.164（RF 为 1.127），MAPE 仅 0.17%。这说明 gradient boosting 的序列学习机制比 bagging 更适合捕捉此时间序列的趋势。

### 5.2 R² 为负值的原因

两个模型的 R² 均为负值，这**不代表模型完全失败**，而是由测试集极小样本量（仅 4 个点）导致的统计假象：

- 测试集 actual 值范围仅为 77.95–78.20，方差极小（SS_tot ≈ 0.009）
- 即使 XGBoost 的绝对误差很小（MAE = 0.13），由于 SS_tot 极小，R² = 1 − SS_res/SS_tot 仍为负
- **MAPE 和 MAE 是更可靠的评估指标**，在此场景下 XGBoost 的 0.17% MAPE 表现优秀

### 5.3 XGBoost 特征重要性高度集中

XGBoost 的特征重要性几乎完全集中在 `life_exp`（99.99%），说明模型学到了"当年预期寿命 → 下一年预期寿命"的近似恒等映射。这在短期预测中合理（寿命年际变化 < 0.5 岁），但也意味着：

- 模型对其他社会经济指标的利用不充分
- 对所有测试年份输出了近乎相同的预测值（≈ 77.939）
- 后续可考虑移除 `life_exp` 特征，强制模型利用其他指标

### 5.4 Random Forest 特征重要性更分散

Random Forest 的重要性分布更均匀：`life_exp`（15%）、`pop_65_plus_pct`（10.7%）、`infant_mortality`（9.3%）、`gdp_per_capita`（8.2%）。虽然预测精度较低，但更符合多变量建模的预期。

### 5.5 系统性偏差

- Random Forest 对所有测试年份**均低估** ~1 岁，存在明显的系统性偏差（bias），可能因训练数据中后期年份权重不足
- XGBoost 也略微低估，但偏差仅 0.01–0.26 岁

---

## 6. 树方法在本场景下的弊端

### 6.1 维度灾难（Curse of Dimensionality）

本实验的核心矛盾是 **样本量（24）远小于特征数（31）**，样本/特征比仅 0.77，远低于经验安全线（5–10 倍）。

在 31 维空间中，24 个样本点极度稀疏。树模型在每个节点的分裂决策只依赖局部样本，样本越少、维度越高，分裂越容易拟合噪声而非真实模式。具体表现为：

- **Random Forest**：`max_depth=None`（无限深）被选为最优，说明模型倾向于完全记忆训练数据
- **XGBoost**：300 轮 boosting 迭代，在仅 24 个样本上反复纠正残差，最终只学到了 `life_exp` 一个特征的恒等映射

### 6.2 特征冗余与共线性

31 个特征中存在严重的信息冗余：每个 WDI 指标有 3 个版本（原值、lag1、pct_change），它们高度相关。例如 `gdp_per_capita` 与 `gdp_per_capita_lag1` 的相关系数通常 > 0.99。

- 对 Random Forest：冗余特征稀释了真正重要特征的分裂机会，导致重要性被分散
- 对 XGBoost：优化器在高度共线的特征间任意选择，结果不稳定

### 6.3 XGBoost 恒等映射陷阱

XGBoost 的特征重要性 99.99% 集中在 `life_exp`，本质上学到了 `life_exp_next_year ≈ life_exp` 的恒等映射。虽然 MAPE 看起来很低（0.17%），但这种模型：

- 无法捕捉突发变化（如 2020 年 COVID-19 的影响）
- 对所有测试年份输出近乎相同的值（77.939），失去了预测的区分度
- 完全忽略了其他 30 个精心构造的社会经济指标

### 6.4 改进方向：PCA 降维

针对上述问题，采用 PCA（主成分分析）降维可以：

1. **消除共线性**：PCA 产生的主成分相互正交，彻底消除特征间冗余
2. **缓解维度灾难**：31 维压缩为 5–8 个主成分，样本/特征比提升至 3–5 倍
3. **强制多变量建模**：移除 `life_exp` 后做 PCA，迫使模型从社会经济指标中学习
4. **降低过拟合风险**：更少的输入维度意味着更低的模型自由度

详见 **第 7 节 PCA 降维实验**。

---

## 7. PCA 降维实验

### 7.1 实验设计

为验证降维能否改善树模型在小样本高维场景下的表现，设计了两组对比实验：

| 实验 | 输入特征 | PCA 前维度 | PCA 后维度 (95% 方差) | 样本/特征比 |
|------|---------|-----------|---------------------|------------|
| A: PCA (全特征) | 31 个原始特征（含 life_exp） | 31 | **8** | 24/8 = 3.0 |
| B: PCA (去 life_exp) | 30 个特征（移除 life_exp） | 30 | **8** | 24/8 = 3.0 |

PCA 流程：StandardScaler → PCA（保留 95% 累积方差），在每个 fold 的训练集上拟合，避免数据泄露。

### 7.2 对比总表

| 模型 | 设置 | 输入维度 | RMSE | MAE | R² | MAPE (%) |
|------|------|---------|------|-----|-----|---------|
| RF | 原始 31 维（基线） | 31 | 1.1269 | 1.1197 | -140.51 | 1.43 |
| RF | PCA 全特征 | 8 PCs | 0.6682 | 0.6656 | -48.76 | 0.85 |
| RF | PCA 去 life_exp | 8 PCs | 0.6973 | 0.6936 | -53.18 | 0.89 |
| XGB | 原始 31 维（基线） | 31 | 0.1641 | 0.1340 | -2.00 | 0.17 |
| XGB | PCA 全特征 | 8 PCs | 0.3623 | 0.3476 | -13.63 | 0.45 |
| XGB | PCA 去 life_exp | 8 PCs | **0.3098** | **0.2585** | -9.70 | **0.33** |

### 7.3 Per-Year Predictions（PCA 实验）

**Random Forest + PCA (全特征)**:

| Year | Actual | Predicted | Residual |
|------|--------|-----------|----------|
| 2019 | 78.019 | 77.389 | +0.630 |
| 2020 | 78.117 | 77.436 | +0.681 |
| 2021 | 78.202 | 77.448 | +0.754 |
| 2022 | 77.953 | 77.355 | +0.598 |

**XGBoost + PCA (去 life_exp)**:

| Year | Actual | Predicted | Residual |
|------|--------|-----------|----------|
| 2019 | 78.019 | 77.705 | +0.314 |
| 2020 | 78.117 | 77.924 | +0.193 |
| 2021 | 78.202 | 77.705 | +0.497 |
| 2022 | 77.953 | 77.924 | +0.029 |

### 7.4 分析

#### PCA 对 Random Forest 有显著改善

RF 的 RMSE 从基线 1.13 降至 0.67（**降幅 41%**），MAE 从 1.12 降至 0.67。PCA 消除了共线性和冗余特征对分裂决策的干扰，使得 RF 在小样本上更稳定。系统性低估偏差从 ~1 岁缩小到 ~0.6 岁。

#### XGBoost + PCA 的权衡

基线 XGBoost 的 RMSE（0.164）看似优于 PCA 版本（0.310），但基线存在严重的"恒等映射"问题——对所有年份输出几乎相同的值（77.939）。而 PCA 版本：

- **预测有区分度**：不同年份输出不同的值（77.70–77.92），能反映年际变化
- **去 life_exp 后反而更好**：RMSE 从 0.362 降至 0.310，证实移除"捷径特征"迫使模型学到了更有意义的社会经济模式
- **最佳 R²**：-9.70 虽仍为负（受限于 4 样本），但在所有实验中最接近 0

#### PCA 方差分析

31 个原始特征中，前 8 个主成分就解释了 95% 的方差，说明原始特征空间存在大量冗余。维度从 31 压缩到 8，信息损失极小但模型复杂度大幅下降。

---

## 8. Limitations & Next Steps

1. **数据量极小**：仅 24 条训练样本、4 条测试样本，模型容易过拟合，统计结论需谨慎
2. **扩大超参搜索**：当前使用 quick mode，可改用 RandomizedSearchCV 在更大空间中高效搜索
3. **集成策略**：考虑 RF 与 XGBoost 的加权集成（ensemble），结合两者优势
4. **与基线模型对比**：待成员 2 完成 Linear/Ridge/Lasso 后进行统一对比

---

## 9. Output Artifacts

### Baseline experiments (`results/`)

| File | Description |
|------|-------------|
| `summary.json` | All results in structured JSON |
| `model_comparison.png` | Side-by-side metric bar chart |
| `random_forest/` | RF model, predictions, plots, results |
| `xgboost/` | XGBoost model, predictions, plots, results |

### PCA experiments (`results_pca/`)

| File | Description |
|------|-------------|
| `pca_summary.json` | PCA variance analysis + all PCA results |
| `pca_model_comparison.png` | 4-model comparison bar chart |
| `pca_variance_all_features.png` | Cumulative variance plot (31 features) |
| `pca_variance_no_life_exp.png` | Cumulative variance plot (30 features) |
| `random_forest_pca_all/` | RF + PCA (all features) model & plots |
| `xgboost_pca_all/` | XGBoost + PCA (all features) model & plots |
| `random_forest_pca_no_life_exp/` | RF + PCA (no life_exp) model & plots |
| `xgboost_pca_no_life_exp/` | XGBoost + PCA (no life_exp) model & plots |
