# 🤖 ML Benchmarking System
**BCD-7213 – Minería de Datos Avanzada · Universidad LEAD · I Cuatrimestre 2026**

> Melany Ramírez · Jason Barrantes · Junior Ramírez  
> Dr. Juan Murillo-Morera

---

## 📋 Descripción

Sistema completo de benchmarking de modelos de Machine Learning con interfaz gráfica en Streamlit. 
Soporta clasificación, regresión y series de tiempo con validación cruzada K-Fold, AUC-ROC, ajuste 
de threshold y manejo de clases desbalanceadas.

## 🏗️ Estructura del Proyecto

```
mlbenchmark/
├── mlbenchmark/                  ← Paquete Python principal
│   ├── __init__.py
│   ├── preprocessing.py          ← División, escalado, codificación
│   ├── balancing.py              ← SMOTE, under-sampling, híbrido
│   ├── metrics.py                ← Métricas de clasificación/regresión/TS
│   ├── validation.py             ← K-Fold, Stratified K-Fold
│   ├── threshold.py              ← Ajuste y optimización de threshold
│   ├── models_classification.py  ← Modelos de clasificación
│   ├── models_regression.py      ← Modelos de regresión
│   ├── models_timeseries.py      ← ARIMA, Holt-Winters, LSTM
│   └── benchmarking.py           ← Orquestador principal
├── app/
│   └── streamlit_app.py          ← Interfaz gráfica Streamlit
├── tests/                        ← Pruebas unitarias
├── setup.py
├── requirements.txt
└── README.md
```

## 🚀 Instalación

```bash
# Clonar o descomprimir el proyecto
cd mlbenchmark

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

## ▶️ Ejecución

### Aplicación Streamlit (GUI)
```bash
streamlit run app/streamlit_app.py
```
Abre tu navegador en `http://localhost:8501`

### Uso como paquete Python
```python
from mlbenchmark.benchmarking import run_benchmark

# Clasificación con SMOTE y K-Fold=5
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

print(result["results"])  # DataFrame con métricas de todos los modelos
```

## 📦 Módulos del Paquete

| Módulo | Funciones Principales |
|--------|----------------------|
| `preprocessing` | `split_data()`, `scale_features()`, `encode_categorical()`, `split_timeseries()` |
| `balancing` | `apply_smote()`, `undersample()`, `apply_combined()`, `check_imbalance()` |
| `metrics` | `classification_metrics()`, `regression_metrics()`, `timeseries_metrics()` |
| `validation` | `kfold_validation()`, `stratified_kfold()`, `manual_kfold()` |
| `threshold` | `apply_threshold()`, `optimize_threshold()`, `threshold_analysis()` |
| `models_classification` | `get_classification_models()`, `predict_classification()` |
| `models_regression` | `get_regression_models()` |
| `models_timeseries` | `HoltWintersModel`, `ARIMAModel`, `LSTMModel`, `get_timeseries_models()` |
| `benchmarking` | `run_benchmark()`, `rank_models()`, `benchmark_classification()` |

## 🤖 Modelos Implementados

### Clasificación
- Logistic Regression, Random Forest, Decision Tree
- SVM (RBF), K-Nearest Neighbors, Naive Bayes
- Gradient Boosting, XGBoost (opcional)

### Regresión
- Ridge, Lasso, Random Forest, Decision Tree
- SVR, K-Nearest Neighbors, Gradient Boosting, XGBoost

### Series de Tiempo
- Holt-Winters (estándar y calibrado)
- ARIMA(1,1,1) y ARIMA calibrado (búsqueda automática de orden)
- LSTM (red neuronal recurrente)

## 📊 Datasets Integrados

| Dataset | Tipo | Muestras | Features |
|---------|------|----------|----------|
| Breast Cancer Wisconsin | Clasificación | 569 | 30 |
| Credit Card Fraud (Simulado) | Clasificación | 10,000 | 20 |
| California Housing | Regresión | 20,640 | 8 |
| Airline Passengers | Series de Tiempo | 144 | — |

## 📖 Referencias

- Hastie et al. (2009). The Elements of Statistical Learning.
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
- Box et al. (2015). Time Series Analysis: Forecasting and Control.
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory.
