# ML-Benchmarking-System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A CRISP-DM–aligned, open-source Python framework for systematic multi-paradigm data mining benchmarking.**

*Developed at Universidad LEAD · Advanced Data Mining (BCD-7312) · IC 2026*

[📄 Scientific Article](#-scientific-article) · [🚀 Quick Start](#-quick-start) · [📊 Results](#-benchmark-results) · [🏗 Architecture](#-system-architecture) · [📖 Citation](#-citation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Scientific Article](#-scientific-article)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Supported Techniques](#-supported-techniques)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Benchmark Results](#-benchmark-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Authors](#-authors)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

**ML-Benchmarking-System** is a reproducible, automated benchmarking suite for data mining and machine learning experiments. It evaluates multiple algorithmic families — classical classifiers, deep neural networks, unsupervised clustering, and association rule mining — under **identical preprocessing and evaluation conditions**, following the full six-phase **CRISP-DM** process model.

The framework was originally developed and validated on the **IBM Telco Customer Churn** prediction problem and has since been extended to serve as a general-purpose benchmarking platform. Its design philosophy prioritizes:

- **Reproducibility** — fixed random seeds, versioned pipelines, exportable configs
- **Fairness** — all models share the same splits, scaling, and SMOTE boundaries
- **Transparency** — every step mirrors a CRISP-DM phase as an auditable module
- **Automation** — metrics, visualizations, and statistical tests generated without manual intervention

> This system is the implementation backbone of the peer-reviewed scientific article listed below.

---

## 📄 Scientific Article

This repository is the official implementation of:

> **"Beyond Classical Churn Prediction: A Comprehensive CRISP-DM Benchmarking of Neural Networks, Clustering, and Association Rules in Telecommunications"**
>
> Jason Jesús Barrantes Sánchez, Melany Ramírez Anchia, Junior Ramírez Carmona
> Universidad LEAD, San José, Costa Rica
> *BCD-7312 – Advanced Data Mining, IC 2026*

### Abstract

Customer churn in the telecommunications sector imposes significant financial costs, as retaining an existing subscriber is estimated to be five to ten times less expensive than acquiring a new one. This paper addresses the gap in multi-paradigm comparisons through a rigorous benchmarking study guided by the full six-phase CRISP-DM methodology, applied to the IBM Telco Customer Churn dataset (*n* = 7,043 records, 21 features). Four classical classifiers and five neural network architectures were evaluated using Accuracy, F1-score, and ROC-AUC, complemented by 5-fold cross-validation and a Wilcoxon Signed-Rank statistical test. K-Means clustering (*k* = 3) revealed a high-risk segment with a **44.18% churn rate**, and Apriori association rule mining identified short tenure and month-to-month contracts as the highest-lift churn predictors (**lift = 1.75**). Neural architectures consistently outperform classical classifiers on minority-class F1 (best: Deep+Dropout, **F1 = 0.787**), while classical models attain competitive ROC-AUC (Logistic Regression: **0.845**).

### Research Questions Addressed

| # | Research Question |
|---|---|
| RQ1 | Which classification algorithm (classical or neural) achieves the best churn predictive performance? |
| RQ2 | Do K-Means customer clusters exhibit statistically distinct churn rates? |
| RQ3 | Which Apriori rules surface the highest lift for `Churn=Yes`? |
| RQ4 | Is there a statistically significant difference in CV F1 between the best neural and classical models? |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔄 **Multi-paradigm** | Classical ML, Deep Learning, Clustering, and Association Rules in one pipeline |
| ⚖️ **Fair comparison** | Identical preprocessing, splits, and folds for every model |
| 📐 **CRISP-DM aligned** | Codebase mirrors the 6 CRISP-DM phases as discrete, auditable modules |
| 📊 **Auto-reporting** | ROC curves, confusion matrices, boxplots, radar charts, lift bar charts — all auto-generated |
| 🧪 **Statistical validation** | Wilcoxon Signed-Rank Test applied programmatically to CV F1 distributions |
| 🔁 **Reproducible** | Fixed random seeds + exportable YAML configs guarantee identical runs |
| 🧬 **SMOTE-safe** | Oversampling applied strictly within training folds — zero leakage to test |
| 🌐 **Interactive UI** | Plotly-powered dashboards for results exploration |
| 📤 **Export-ready** | CSV reports, LaTeX-formatted tables, and Plotly HTML charts |

---

## 🏗 System Architecture

```
ML-Benchmarking-System/
│
├── Phase 1 · Business Understanding   →  config/business_config.yaml
├── Phase 2 · Data Understanding       →  src/eda/
│     ├── descriptive_stats.py
│     ├── class_distribution.py
│     └── pairplot.py
│
├── Phase 3 · Data Preparation         →  src/preprocessing/
│     ├── imputer.py
│     ├── encoder.py
│     ├── scaler.py
│     ├── smote_pipeline.py
│     └── apriori_discretizer.py
│
├── Phase 4 · Modeling                 →  src/models/
│     ├── classical/
│     │     ├── logistic_regression.py
│     │     ├── random_forest.py
│     │     ├── gradient_boosting.py
│     │     └── xgboost_model.py
│     ├── neural/
│     │     ├── mlp_sklearn.py
│     │     ├── feedforward_keras.py
│     │     ├── deep_dropout_keras.py
│     │     ├── cnn1d_keras.py
│     │     └── autoencoder_keras.py
│     ├── clustering/
│     │     └── kmeans.py
│     └── association/
│           └── apriori.py
│
├── Phase 5 · Evaluation               →  src/evaluation/
│     ├── metrics.py
│     ├── cross_validation.py
│     ├── wilcoxon_test.py
│     └── cluster_profiler.py
│
├── Phase 6 · Deployment               →  src/reporting/
│     ├── visualizations.py
│     └── export.py
│
├── data/                              →  datasets (not tracked by git)
├── results/                           →  auto-generated outputs
├── notebooks/                         →  Jupyter exploration notebooks
├── paper/                             →  LaTeX source of the scientific article
│     ├── main.tex
│     └── figures/
└── main.py                            →  single entrypoint to run full benchmark
```

---

## 🤖 Supported Techniques

### Classical Classifiers
| Model | Library | Notes |
|---|---|---|
| Logistic Regression | scikit-learn | L2 regularization, solver=lbfgs |
| Random Forest | scikit-learn | n_estimators=200, class_weight=balanced |
| Gradient Boosting | scikit-learn | n_estimators=200, learning_rate=0.05 |
| XGBoost | xgboost | scale_pos_weight tuned per fold |

### Neural Network Architectures
| Architecture | Framework | Description |
|---|---|---|
| MLP | scikit-learn | 2 hidden layers (128, 64), ReLU, Adam |
| Feedforward | Keras / TF | 2 hidden layers, BatchNorm, sigmoid output |
| Deep + Dropout | Keras / TF | 3 hidden layers, Dropout(0.3–0.5), EarlyStopping |
| CNN-1D | Keras / TF | Treats feature vector as 1D temporal sequence |
| Autoencoder | Keras / TF | Trained on non-churn only; anomaly threshold = 0.256 |

### Unsupervised
| Technique | Details |
|---|---|
| K-Means | k ∈ {2, 3, 4}, Elbow + Silhouette selection, PCA 2D visualization |

### Association Rules
| Algorithm | Config |
|---|---|
| Apriori | min_support=0.05, min_confidence=0.25, max_len=3, consequent=Churn=Yes |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/sklinderton/ML-Benchmarking-System.git
cd ML-Benchmarking-System

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the Telco dataset in data/
#    Download from: https://www.kaggle.com/blastchar/telco-customer-churn
cp ~/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv data/

# 5. Run the full benchmark
python main.py --config config/default.yaml

# Results are saved to results/ automatically
```

---

## 📦 Installation

### Requirements

```
Python >= 3.9
tensorflow >= 2.10
scikit-learn >= 1.2
xgboost >= 1.7
mlxtend >= 0.22          # Apriori
imbalanced-learn >= 0.10 # SMOTE
plotly >= 5.14
pandas >= 1.5
numpy >= 1.23
scipy >= 1.10
matplotlib >= 3.7
seaborn >= 0.12
pyyaml >= 6.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

### Optional — GPU support (for Keras models)

```bash
pip install tensorflow[and-cuda]    # TF 2.12+
```

---

## 🧑‍💻 Usage

### Run the full pipeline

```bash
python main.py --config config/default.yaml
```

### Run individual modules

```python
from src.preprocessing.smote_pipeline import build_pipeline
from src.models.classical.xgboost_model import run_xgboost
from src.models.neural.deep_dropout_keras import run_deep_dropout
from src.models.clustering.kmeans import run_kmeans
from src.models.association.apriori import run_apriori
from src.evaluation.wilcoxon_test import compare_models

# Load and preprocess
X_train, X_test, y_train, y_test = build_pipeline("data/telco.csv")

# Classical
xgb_results = run_xgboost(X_train, X_test, y_train, y_test)

# Neural
nn_results = run_deep_dropout(X_train, X_test, y_train, y_test)

# Unsupervised
cluster_profiles = run_kmeans(X_train, k=3)

# Rules
rules_df = run_apriori(X_train, y_train,
                       min_support=0.05,
                       min_confidence=0.25)

# Statistical test
wilcoxon_result = compare_models(xgb_results["cv_f1"],
                                 nn_results["cv_f1"])
print(f"W={wilcoxon_result.statistic}, p={wilcoxon_result.pvalue:.4f}")
```

### Configuration file (`config/default.yaml`)

```yaml
data:
  path: data/telco.csv
  target: Churn
  test_size: 0.30
  random_state: 42

preprocessing:
  smote_strategy: 0.5
  scaler: standard
  encoding: onehot

cross_validation:
  n_splits: 5
  n_repeats: 3
  scoring: f1

models:
  classical: [logistic_regression, random_forest, gradient_boosting, xgboost]
  neural:    [mlp, feedforward, deep_dropout, cnn1d, autoencoder]
  clustering:
    kmeans:
      k_range: [2, 3, 4]
  association:
    apriori:
      min_support: 0.05
      min_confidence: 0.25
      max_len: 3

output:
  results_dir: results/
  export_csv: true
  export_html: true
```

---

## 📊 Benchmark Results

Results obtained on the **IBM Telco Customer Churn** dataset (*n* = 7,043, 30% held-out test set).

### Classification — Test Set

| Family | Model | Accuracy | Precision | Recall | **F1** | **AUC-ROC** |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Classical | XGBoost | 0.754 | 0.525 | 0.783 | **0.629** | **0.843** |
| Classical | Logistic Regression | 0.680 | 0.448 | 0.893 | 0.597 | **0.845** |
| Classical | Gradient Boosting | 0.774 | 0.597 | 0.460 | 0.520 | 0.826 |
| Classical | Random Forest | 0.717 | 0.481 | 0.829 | 0.608 | 0.819 |
| Neural | MLP (sklearn) | 0.743 | — | — | 0.742 | 0.769 |
| Neural | Feedforward (Keras) | 0.792 | — | — | 0.784 | 0.832 |
| Neural | **Deep+Dropout (Keras)** | 0.791 | — | — | **0.787** | 0.837 |
| Neural | CNN-1D (Keras) | 0.786 | — | — | 0.779 | 0.822 |
| Neural | Autoencoder* | — | — | — | — | — |

> *Autoencoder: unsupervised anomaly detection. Mean reconstruction error = 0.141, threshold = 0.256.  
> Flagged **163 anomalies (11.57%)** in the test set as potential churners.

### Cross-Validation — 5-Fold AUC-ROC (Classical Classifiers)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** | **Std** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| XGBoost | 0.843 | 0.830 | 0.846 | 0.863 | 0.862 | **0.849** | 0.012 |
| Logistic Regression | 0.847 | 0.826 | 0.839 | 0.858 | 0.851 | 0.844 | 0.011 |
| Gradient Boosting | 0.841 | 0.830 | 0.845 | 0.858 | 0.864 | 0.848 | 0.012 |
| Random Forest | 0.837 | 0.829 | 0.843 | 0.846 | 0.857 | 0.842 | 0.009 |

### Statistical Validation — Wilcoxon Signed-Rank Test

| Model A | Model B | F1 (A) | F1 (B) | W | **p-value** | Reject H₀? |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | Gradient Boosting | 0.599 ± 0.023 | 0.587 ± 0.021 | 21.0 | **0.026** | ✅ Yes (α=0.05) |

### K-Means Clustering — Customer Segment Profiles (*k* = 3)

| Cluster | Label | Size | **Churn %** | Avg Tenure | Avg Monthly Charges | Partner % |
|---|---|:---:|:---:|:---:|:---:|:---:|
| C0 | High-Value Loyal | 2,362 | 15.33% | 55.07 mo | $89.60 | 71.1% |
| C1 | **High-Risk Early** | 3,155 | **44.18%** | 16.26 mo | $67.30 | 31.2% |
| C2 | Low-Spend Stable | 1,526 | 7.40% | 30.55 mo | $21.08 | 48.4% |

> Chi-square test confirms strong association between cluster and churn: χ² = 1,841.2, df = 2, **p < 0.001**

### Top-10 Association Rules (Apriori, consequent = `Churn=Yes`)

| Antecedent | Support | Confidence | **Lift** |
|---|:---:|:---:|:---:|
| `tenure=Bajo` | 0.156 | 0.464 | **1.747** |
| `Contract=Month-to-month` | 0.235 | 0.427 | **1.609** |
| `InternetService=Fiber optic` | 0.184 | 0.419 | 1.579 |
| `OnlineSecurity=No` | 0.207 | 0.418 | 1.574 |
| `SeniorCitizen=Yes` | 0.068 | 0.417 | 1.571 |
| `TechSupport=No` | 0.205 | 0.416 | 1.569 |
| `OnlineBackup=No` | 0.175 | 0.399 | 1.505 |
| `DeviceProtection=No` | 0.172 | 0.391 | 1.475 |
| `MonthlyCharges=Alto` | 0.114 | 0.341 | 1.285 |
| `StreamingMovies=No` | 0.133 | 0.337 | 1.269 |

---

## 📁 Dataset

This framework was benchmarked on the **IBM Telco Customer Churn** dataset:

| Property | Value |
|---|---|
| Records | 7,043 subscribers |
| Features | 20 input + 1 target (`Churn`) |
| Class balance | 73.5% No / 26.5% Yes |
| Feature types | Demographics, services, contract, billing |
| Source | [Kaggle — IBM Sample Dataset](https://www.kaggle.com/blastchar/telco-customer-churn) |

> The dataset is **not included** in this repository due to licensing. Download it from the link above and place the CSV in `data/`.

---

## 📂 Project Structure

```
ML-Benchmarking-System/
├── config/
│   └── default.yaml                  # Main configuration file
├── data/                             # Place your CSV here (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_neural_networks.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_association_rules.ipynb
├── paper/
│   ├── main.tex                      # Full IEEE-format scientific article
│   └── figures/                      # Auto-generated plots used in the paper
├── results/
│   ├── metrics/                      # CSV exports per model
│   ├── plots/                        # PNG/HTML visualizations
│   └── reports/                      # Full benchmark summary
├── src/
│   ├── eda/
│   ├── preprocessing/
│   ├── models/
│   │   ├── classical/
│   │   ├── neural/
│   │   ├── clustering/
│   │   └── association/
│   ├── evaluation/
│   └── reporting/
├── main.py                           # Pipeline entrypoint
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 👥 Authors

| Name | Role | Email | Institution |
|---|---|---|---|
| **Jason Jesús Barrantes Sánchez** | Lead Developer & ML Engineer | jason.barrantes@ulead.ac.cr | Universidad LEAD |
| **Melany Ramírez Anchia** | Data Analyst & Visualization | melany.ramirez@ulead.ac.cr | Universidad LEAD |
| **Junior Ramírez Carmona** | Neural Networks & Association Rules | junior.ramirez@ulead.ac.cr | Universidad LEAD |

Supervised by **Dr. Juan Murillo-Morera** — Advanced Data Mining, BCD-7312, IC 2026.

---

## 📖 Citation

If you use this framework or reference the associated scientific article, please cite:

```bibtex
@inproceedings{barrantes2026churn,
  author    = {Barrantes Sánchez, Jason Jesús and
               Ramírez Anchia, Melany and
               Ramírez Carmona, Junior},
  title     = {Beyond Classical Churn Prediction: A Comprehensive {CRISP-DM}
               Benchmarking of Neural Networks, Clustering, and Association Rules
               in Telecommunications},
  booktitle = {BCD-7312 -- Advanced Data Mining, Universidad LEAD},
  year      = {2026},
  address   = {San José, Costa Rica},
  url       = {https://github.com/sklinderton/ML-Benchmarking-System}
}
```

Plain-text citation:

> J. J. Barrantes Sánchez, M. Ramírez Anchia, and J. Ramírez Carmona, "Beyond Classical Churn Prediction: A Comprehensive CRISP-DM Benchmarking of Neural Networks, Clustering, and Association Rules in Telecommunications," Universidad LEAD, San José, Costa Rica, 2026. [Online]. Available: https://github.com/sklinderton/ML-Benchmarking-System

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Resources

- 📄 [Scientific Article (PDF)](paper/main.tex) — Full IEEE-format paper
- 📊 [IBM Telco Dataset](https://www.kaggle.com/blastchar/telco-customer-churn) — Kaggle
- 📐 [CRISP-DM Reference](https://the-modeling-agency.com/crisp-dm.pdf) — Chapman et al., 2000
- 🧠 [Deep Learning Book](https://www.deeplearningbook.org) — Goodfellow, Bengio & Courville
- 🌿 [Overleaf Project](https://www.overleaf.com) — LaTeX source (invite-only)

---

<div align="center">

Made with ❤️ at **Universidad LEAD** · San José, Costa Rica · 2025–2026

</div>
