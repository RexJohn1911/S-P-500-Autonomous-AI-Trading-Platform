# Machine Learning & AI Model Specification

## 1. Overview
The platform leverages an ensemble of baseline machine learning algorithms, deep neural sequence architectures, and unsupervised regime models to generate predictive directional signals for S&P 500 equities.

---

## 2. Baseline ML Models (Phase 07)
Trained using chronological walk-forward splits to forecast 5-day forward price movement:
- **Logistic Regression**: Linear benchmark with L2 regularization.
- **Random Forest**: Ensemble of decision trees with bagging.
- **XGBoost**: Extreme Gradient Boosting with depth constraints.
- **LightGBM**: Fast histogram-based gradient boosting.

### Validated Empirical Results (Phase 07)
*Measured on historical test sets under strict zero-leakage cross-validation:*

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 43.75% | 0.5000 | 0.5000 | 0.5000 | 0.4921 | 0.6782 |
| **Random Forest** | 54.69% | 0.5593 | 0.9167 | 0.6947 | 0.5248 | 0.5710 |
| **XGBoost** | 56.25% | 0.5690 | 0.9167 | 0.7021 | 0.5774 | 0.6564 |
| **LightGBM** | 56.25% | 0.5909 | 0.7222 | 0.6500 | 0.5699 | 0.6303 |

---

## 3. Advanced AI Models (Phase 08)
- **Feed-Forward MLP**: Dense multi-layer neural network with Dropout and Batch Normalization.
- **LSTM (Long Short-Term Memory)**: Recurrent neural network capturing 15-day sequential market patterns.
- **Temporal Multi-Head Attention Transformer**: Self-attention mechanism learning cross-temporal feature interactions.

### Validated Transformer Results (Phase 08)
- **Precision**: **76.92%**
- **PR-AUC**: **0.8249**
- **Directional Spread**: **+0.52%**

---

## 4. Market Regime Detection (Phase 09)
Unsupervised clustering and probabilistic state-space modeling:
- **K-Means Clustering**: Partitions market states based on 20-day returns, rolling volatility, and moving average divergences.
- **Gaussian Mixture Models (GMM)**: Soft probabilistic regime assignments.
- **Hidden Markov Models (HMM)**: Sequential regime transitions estimating Bull, Bear, and Sideways states.
