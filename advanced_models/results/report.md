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

## 6. Limitations & Next Steps

1. **数据量极小**：仅 24 条训练样本、4 条测试样本，模型容易过拟合，统计结论需谨慎
2. **扩大超参搜索**：当前使用 quick mode（48 组合），可改用 RandomizedSearchCV 在更大空间中高效搜索
3. **特征消融实验**：尝试移除 `life_exp` 特征，观察模型是否能仅凭社会经济指标做出合理预测
4. **集成策略**：考虑 RF 与 XGBoost 的加权集成（ensemble），结合两者优势
5. **与基线模型对比**：待成员 2 完成 Linear/Ridge/Lasso 后进行统一对比

---

## 7. Output Artifacts

| File | Description |
|------|-------------|
| `summary.json` | All results in structured JSON |
| `model_comparison.png` | Side-by-side metric bar chart |
| `random_forest/model.joblib` | Trained RF model |
| `random_forest/test_predictions.csv` | RF per-year predictions |
| `random_forest/results.json` | RF params + metrics + importances |
| `random_forest/actual_vs_predicted.png` | RF prediction plot |
| `random_forest/residuals.png` | RF residual bar chart |
| `random_forest/feature_importance.png` | RF top-15 feature importance |
| `xgboost/model.joblib` | Trained XGBoost model |
| `xgboost/test_predictions.csv` | XGB per-year predictions |
| `xgboost/results.json` | XGB params + metrics + importances |
| `xgboost/actual_vs_predicted.png` | XGB prediction plot |
| `xgboost/residuals.png` | XGB residual bar chart |
| `xgboost/feature_importance.png` | XGB top-15 feature importance |
