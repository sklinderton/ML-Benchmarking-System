```markdown
# ML Benchmarking System

> Melany Ramírez  

## Description

Complete benchmarking system for Machine Learning models with a graphical interface built in Streamlit.  
Supports classification, regression, and time series with K-Fold cross-validation, AUC-ROC, threshold adjustment, and imbalanced class handling.

---

## Project Structure

```

mlbenchmark/
├── mlbenchmark/                  ← Main Python package
│   ├── **init**.py
│   ├── preprocessing.py          ← Data splitting, scaling, encoding
│   ├── balancing.py              ← SMOTE, under-sampling, hybrid
│   ├── metrics.py                ← Classification/Regression/TS metrics
│   ├── validation.py             ← K-Fold, Stratified K-Fold
│   ├── threshold.py              ← Threshold adjustment and optimization
│   ├── models_classification.py  ← Classification models
│   ├── models_regression.py      ← Regression models
│   ├── models_timeseries.py      ← ARIMA, Holt-Winters, LSTM
│   └── benchmarking.py           ← Main orchestrator
├── app/
│   └── streamlit_app.py          ← Streamlit GUI
├── tests/                        ← Unit tests
├── setup.py
├── requirements.txt
└── README.md

````

---

## Installation

```bash
# Clone or extract the project
cd mlbenchmark

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
````

---

## Execution

### Streamlit Application (GUI)

```bash
streamlit run app/streamlit_app.py
```

Open your browser at:

```
http://localhost:8501
```

### Using as a Python Package

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

* Logistic Regression
* Random Forest
* Decision Tree
* SVM (RBF)
* K-Nearest Neighbors
* Naive Bayes
* Gradient Boosting
* XGBoost (optional)

### Regression

* Ridge
* Lasso
* Random Forest
* Decision Tree
* SVR
* K-Nearest Neighbors
* Gradient Boosting
* XGBoost

### Time Series

* Holt-Winters (standard and calibrated)
* ARIMA(1,1,1) and calibrated ARIMA (automatic order search)
* LSTM (Recurrent Neural Network)

---

## Integrated Datasets

| Dataset                       | Type           | Samples | Features |
| ----------------------------- | -------------- | ------- | -------- |
| Breast Cancer Wisconsin       | Classification | 569     | 30       |
| Credit Card Fraud (Simulated) | Classification | 10,000  | 20       |
| California Housing            | Regression     | 20,640  | 8        |
| Airline Passengers            | Time Series    | 144     | —        |

```
```
