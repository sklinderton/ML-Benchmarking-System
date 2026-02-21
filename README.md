# ML Benchmarking System

**BCD-7213 – Advanced Data Mining · Universidad LEAD · Term I 2026**

> Melany Ramírez · Jason Barrantes · Junior Ramírez
> Dr. Juan Murillo-Morera

---

## Description

Complete benchmarking system for Machine Learning models with a Streamlit graphical interface.
Supports classification, regression, and time series with K-Fold cross-validation, AUC-ROC,
threshold tuning, and handling of imbalanced classes.

---

## Project Structure

```
mlbenchmark/
├── mlbenchmark/                  ← Main Python package
│   ├── __init__.py
│   ├── preprocessing.py          ← Splitting, scaling, encoding
│   ├── balancing.py              ← SMOTE, under-sampling, hybrid methods
│   ├── metrics.py                ← Classification/regression/TS metrics
│   ├── validation.py             ← K-Fold, Stratified K-Fold
│   ├── threshold.py              ← Threshold tuning and optimization
│   ├── models_classification.py  ← Classification models
│   ├── models_regression.py      ← Regression models
│   ├── models_timeseries.py      ← ARIMA, Holt-Winters, LSTM
│   └── benchmarking.py           ← Main orchestrator
├── app/
│   └── streamlit_app.py          ← Streamlit graphical interface
├── tests/                        ← Unit tests
├── setup.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone or unzip the project
cd mlbenchmark

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

---

## Execution

### Streamlit Application (GUI)

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`

### Using as a Python package

```python
from mlbenchmark.benchmarking import run_benchmark

# Classification with SMOTE and K-Fold=5
result = run_benchmark(
    problem_type="classification",
    X=X.values,
    y=y.values,
    test_size=0.3,
    cv_folds=5,
    threshold=0.5,
    balancing_technique="smote",
    scale=True,
)

print(result["results"])  # DataFrame with metrics for all models
```

---

## Package Modules

| Module                  | Main Functions                                                                   |
| ----------------------- | -------------------------------------------------------------------------------- |
| `preprocessing`         | `split_data()`, `scale_features()`, `encode_categorical()`, `split_timeseries()` |
| `balancing`             | `apply_smote()`, `undersample()`, `apply_combined()`, `check_imbalance()`        |
| `metrics`               | `classification_metrics()`, `regression_metrics()`, `timeseries_metrics()`       |
| `validation`            | `kfold_validation()`, `stratified_kfold()`, `manual_kfold()`                     |
| `threshold`             | `apply_threshold()`, `optimize_threshold()`, `threshold_analysis()`              |
| `models_classification` | `get_classification_models()`, `predict_classification()`                        |
| `models_regression`     | `get_regression_models()`                                                        |
| `models_timeseries`     | `HoltWintersModel`, `ARIMAModel`, `LSTMModel`, `get_timeseries_models()`         |
| `benchmarking`          | `run_benchmark()`, `rank_models()`, `benchmark_classification()`                 |

---

## Implemented Models

### Classification

* Logistic Regression, Random Forest, Decision Tree
* SVM (RBF), K-Nearest Neighbors, Naive Bayes
* Gradient Boosting, XGBoost (optional)

### Regression

* Ridge, Lasso, Random Forest, Decision Tree
* SVR, K-Nearest Neighbors, Gradient Boosting, XGBoost

### Time Series

* Holt-Winters (standard and tuned)
* ARIMA(1,1,1) and tuned ARIMA (automatic order search)
* LSTM (recurrent neural network)

---

## Included Datasets

| Dataset                       | Type           | Samples | Features |
| ----------------------------- | -------------- | ------- | -------- |
| Breast Cancer Wisconsin       | Classification | 569     | 30       |
| Credit Card Fraud (Simulated) | Classification | 10,000  | 20       |
| California Housing            | Regression     | 20,640  | 8        |
| Airline Passengers            | Time Series    | 144     | —        |

---

## References

* Hastie et al. (2009). *The Elements of Statistical Learning.*
* Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.*
* Box et al. (2015). *Time Series Analysis: Forecasting and Control.*
* Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.*
