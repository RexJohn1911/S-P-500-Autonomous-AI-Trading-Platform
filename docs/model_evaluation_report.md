# S&P 500 AI Trading System — Model Accuracy & Out-of-Sample Validation Report

**Evaluation Timestamp:** `2026-09-08T10:35:39.066468+00:00`  
**Dataset Identifier:** `AAPL` (Daily Bars)  
**Unseen Dataset Status:** **INDEPENDENT OUT-OF-SAMPLE DATASET (CROSS-ASSET HOLDOUT)**  

---

## 1. Executive Summary
This evaluation report presents a rigorous, zero-lookahead statistical and out-of-sample performance assessment of the trained machine learning ensemble and individual constituent models. All preprocessing transforms were strictly fitted on historical training observations, and evaluation was performed on chronological test data.

---

## 2. Dataset & Evaluation Period
| Property | Value |
|---|---|
| Target Asset | `AAPL` |
| Total Bar Count | `523` bars |
| Historical Date Range | `2023-10-06` to `2024-12-25` |
| Out-of-Sample Test Range | `2024-09-27` to `2024-12-25` |
| Holdout Sample Count | `64` observations |
| Data Source | `Local Validated Daily Bar Store` |
| Used for Training | `0` |
| Used for Hyperparameter Tuning | `0` |

---

## 3. Target Definition & Feature Set
- **Prediction Target:** `future_return_5d_pos` (Binary classification: `1` if 5-day forward return > `+0.50%`, `0` otherwise)
- **Prediction Horizon:** `5 trading days`
- **Feature Suite (16 Causal Technical Features):**
  `atr_14`, `ema_20`, `ema_200`, `ema_50`, `price_vs_sma_20`, `price_vs_sma_200`, `price_vs_sma_50`, `return_1d`, `return_20d`, `return_5d`, `rolling_volatility_20d`, `sma_20`, `sma_200`, `sma_50`, `volume_change_1d`, `volume_zscore_20d`

---

## 4. Classification Performance Comparison
| Model | Accuracy | Baseline Acc | Improvement | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|---|---|---|
| **logistic_regression** | **42.19%** | 54.69% | -12.50% | 0.4355 | 0.9310 | 0.5934 | 0.3153 | 0.3632 | 1.7475 | 0.5023 |
| **random_forest** | **43.75%** | 54.69% | -10.94% | 0.4426 | 0.9310 | 0.6000 | 0.5655 | 0.5216 | 0.7554 | 0.2777 |
| **xgboost** | **43.75%** | 54.69% | -10.94% | 0.4444 | 0.9655 | 0.6087 | 0.5635 | 0.6571 | 0.7784 | 0.2916 |
| **lightgbm** | **40.62%** | 54.69% | -14.06% | 0.4262 | 0.8966 | 0.5778 | 0.5197 | 0.5575 | 0.8532 | 0.3217 |
| **mlp** | **43.75%** | 54.69% | -10.94% | 0.4444 | 0.9655 | 0.6087 | 0.4069 | 0.4362 | 0.9059 | 0.3395 |
| **lstm** | **32.00%** | 68.00% | -36.00% | 0.3200 | 1.0000 | 0.4848 | 0.2316 | 0.2269 | 1.1347 | 0.4375 |
| **transformer** | **32.00%** | 68.00% | -36.00% | 0.3200 | 1.0000 | 0.4848 | 0.5551 | 0.6246 | 1.1465 | 0.4413 |
| **ENSEMBLE (Weighted)** | **43.75%** | 54.69% | -10.94% | 0.4444 | 0.9655 | 0.6087 | 0.4355 | 0.5044 | 0.9109 | 0.3462 |

---

## 5. Confusion Matrices (Out-of-Sample Test)
| Model | True Positive | True Negative | False Positive | False Negative | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|---|
| **logistic_regression** | 27 | 0 | 35 | 2 | 93.10% | 0.00% | 43.55% |
| **random_forest** | 27 | 1 | 34 | 2 | 93.10% | 2.86% | 44.26% |
| **xgboost** | 28 | 0 | 35 | 1 | 96.55% | 0.00% | 44.44% |
| **lightgbm** | 26 | 0 | 35 | 3 | 89.66% | 0.00% | 42.62% |
| **mlp** | 28 | 0 | 35 | 1 | 96.55% | 0.00% | 44.44% |
| **lstm** | 16 | 0 | 34 | 0 | 100.00% | 0.00% | 32.00% |
| **transformer** | 16 | 0 | 34 | 0 | 100.00% | 0.00% | 32.00% |
| **ENSEMBLE** | 28 | 0 | 35 | 1 | 96.55% | 0.00% | 44.44% |

---

## 6. Probability Calibration & Reliability
- **Ensemble Brier Score:** `0.3462`
- **Expected Calibration Error (ECE):** `0.3413`

| Confidence Bucket | Count | Mean Predicted Prob | Actual Positive Rate | Status |
|---|---|---|---|---|
| `0.50–0.60` | 4 | 54.18% | 100.00% | VALID |
| `0.60–0.70` | 9 | 67.81% | 44.44% | VALID |
| `0.70–0.80` | 43 | 74.75% | 34.88% | VALID |
| `0.80–0.90` | 7 | 82.36% | 71.43% | VALID |
| `0.90–1.00` | 0 | N/A | N/A | INSUFFICIENT SAMPLE |

---

## 7. Forward Return Analysis (5-Day Realized Returns)
| Model | Mean Long Return | Mean Short Return | Return Spread | Median Long Ret | Median Short Ret |
|---|---|---|---|---|---|
| **logistic_regression** | +0.61% | +1.64% | **-1.03%** | +0.21% | +1.64% |
| **random_forest** | +0.57% | +2.20% | **-1.64%** | +0.23% | +2.62% |
| **xgboost** | +0.61% | +2.62% | **-2.01%** | +0.23% | +2.62% |
| **lightgbm** | +0.49% | +3.67% | **-3.18%** | +0.20% | +3.33% |
| **mlp** | +0.61% | +2.62% | **-2.01%** | +0.23% | +2.62% |
| **lstm** | -0.06% | N/A | **N/A** | -0.46% | N/A |
| **transformer** | -0.06% | N/A | **N/A** | -0.46% | N/A |
| **ENSEMBLE** | +0.61% | +2.62% | **-2.01%** | +0.23% | +2.62% |

---

## 8. Signal Engine Synthesis
- **Total Signal Directives:** `64`
- **LONG Conviction:** `58` (90.6%)
- **SHORT Conviction:** `0` (0.0%)
- **FLAT (Neutral / Gated):** `6` (9.4%)
- **Average Conviction Confidence:** `0.46`
- **Average Model Agreement:** `98.53%`

### Reason Codes Breakdown:
- `NEUTRAL_SCORE`: 5
- `LOW_CONFIDENCE`: 1
- `BULLISH_ENSEMBLE_SCORE`: 58
- `MODEL_CONSENSUS`: 58
- `REGIME_UNKNOWN_DEFAULT`: 58

---

## 9. Out-of-Sample Trading Backtest vs Benchmark
| Metric | Strategy (Ensemble Signals) | Benchmark (SPY Buy & Hold) |
|---|---|---|
| **Total Return** | **+1.27%** | +8.18% |
| **CAGR** | **+4.72%** | +33.27% |
| **Annualized Volatility** | `10.90%` | `18.00%` |
| **Sharpe Ratio** | **0.483** | 1.313 |
| **Maximum Drawdown** | **6.29%** | 12.31% |
| **Alpha (Annualized)** | **-0.0845** | N/A |
| **Beta** | `0.40` | 1.00 |
| **Correlation** | `0.89` | 1.00 |
| **Total Executed Trades** | `32` | 1 |
| **Win Rate** | `81.2%` | N/A |
| **Total Transaction Friction** | `$118.62` | $0.00 |

---

## 10. Walk-Forward Cross-Validation Analysis
| Fold | Train Period | Test Period | Train Samples | Test Samples | Accuracy | F1 Score | ROC-AUC | Return Spread |
|---|---|---|---|---|---|---|---|---|
| Fold 1 | 2023-10-06..2024-01-02 | 2024-01-03..2024-03-29 | 63 | 63 | 49.21% | 0.4483 | 0.4819 | -0.82% |
| Fold 2 | 2023-10-06..2024-03-29 | 2024-04-01..2024-06-26 | 126 | 63 | 60.32% | 0.5283 | 0.6562 | +2.26% |
| Fold 3 | 2023-10-06..2024-06-26 | 2024-06-27..2024-09-23 | 189 | 63 | 58.73% | 0.2778 | 0.7859 | +3.93% |
| Fold 4 | 2023-10-06..2024-09-23 | 2024-09-24..2024-12-19 | 252 | 63 | 42.86% | 0.5714 | 0.6947 | +2.04% |

- **Mean Accuracy across Folds:** `52.78%` (Std: `7.13%`)
- **Median Accuracy:** `53.97%` [Min: `42.86%`, Max: `60.32%`]
- **Mean F1 Score:** `0.4565`

---

## 11. Data Leakage & Reproducibility Verification
| Validation Check | Status | Verification Detail |
|---|---|---|
| `future_rows_cannot_influence_current_features` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `training_preprocessing_not_fitted_on_test_data` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `test_data_remains_chronological` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `labels_use_only_future_observations` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `feature_timestamps_prior_or_equal_to_prediction` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `prediction_timestamp_strictly_prior_to_forward_evaluation` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `final_test_data_not_used_for_training` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `final_test_data_not_used_for_threshold_tuning` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `symbol_data_does_not_leak_across_securities` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `walk_forward_folds_preserve_chronology` | **PASS** | Verified by zero-lookahead temporal boundary suite |
| `tabular_models_deterministic` | **PASS** | Consecutive inference passes yield bitwise identical outputs |
| `predictions_reproducible` | **PASS** | Consecutive inference passes yield bitwise identical outputs |
| `random_seed_locked` | **PASS** | Consecutive inference passes yield bitwise identical outputs |

---

## 12. Scientific Limitations & Conclusion
- Market regimes shift dynamically over time; static ensemble weights require ongoing walk-forward monitoring.
- Historical simulated performance does not guarantee future live execution returns under changing liquidity conditions.
- High majority class skew in bull markets elevates baseline accuracy; models must be evaluated against majority baseline.

### Conclusion
Model evaluation completed successfully across 7 models on AAPL daily dataset. Ensemble achieved 43.75% accuracy vs 54.69% majority baseline with a forward return spread of -2.01%.
