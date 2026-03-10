<div align="center">

# 🤖 ML Benchmarking System

### Sistema de Benchmarking de Modelos de Machine Learning  
### con Validación Cruzada K-Fold, AUC-ROC y Manejo de Clases Desbalanceadas

---

**Universidad LEAD — Escuela de Ciencias de Datos**  
**BCD-7213 · Minería de Datos Avanzada · I Cuatrimestre 2026**

| | |
|---|---|
| **Profesor** | Dr. Juan Murillo-Morera |
| **Integrantes** | Melany Ramírez · Jason Barrantes · Junior Ramírez |
| **Entrega** | Caso de Estudio 1 — 14 de marzo de 2026 |

</div>

---

## 📋 Resumen

Sistema completo de benchmarking para modelos de machine learning con interfaz gráfica desarrollada en Streamlit. Soporta problemas de **clasificación**, **regresión** y **series de tiempo**, e implementa validación cruzada K-Fold estratificada, métricas AUC-ROC, ajuste de umbral de decisión (threshold) y manejo de clases desbalanceadas mediante SMOTE, under-sampling y estrategia híbrida.

**Palabras clave:** Machine Learning · Benchmarking · K-Fold Cross-Validation · AUC-ROC · Clases Desbalanceadas · Streamlit · Series de Tiempo

---

## 🗂️ Estructura del Proyecto

```
mlbenchmark/
├── mlbenchmark/                    ← Paquete principal Python
│   ├── __init__.py
│   ├── preprocessing.py            ← División, escalado, encoding
│   ├── balancing.py                ← SMOTE, under-sampling, híbrido
│   ├── metrics.py                  ← Métricas clasificación/regresión/TS
│   ├── validation.py               ← K-Fold, Stratified K-Fold
│   ├── threshold.py                ← Ajuste y optimización de threshold
│   ├── models_classification.py    ← Modelos de clasificación
│   ├── models_regression.py        ← Modelos de regresión
│   ├── models_timeseries.py        ← ARIMA, Holt-Winters, LSTM
│   ├── eda.py                      ← Clase analisisEDA (CRISP-DM)
│   └── benchmarking.py             ← Orquestador principal
├── app/
│   └── streamlit_app.py            ← Interfaz gráfica Streamlit
├── tests/                          ← Pruebas unitarias
├── setup.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

### Requisitos

- Python ≥ 3.9
- pip

### Pasos

```bash
# 1. Clonar o extraer el proyecto
cd mlbenchmark

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar el paquete en modo desarrollo
pip install -e .
```

---

## 🚀 Ejecución

### Interfaz Gráfica (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Abrir en el navegador: **http://localhost:8501**

### Uso como Paquete Python

```python
from mlbenchmark.benchmarking import run_benchmark

# Clasificación con SMOTE y K-Fold = 5
result = run_benchmark(
    problem_type="classification",
    X=X,                          # DataFrame o ndarray
    y=y,
    test_size=0.3,
    cv_folds=5,
    threshold=0.5,
    balancing_technique="smote",
    scale=True,
)

print(result["results"])          # DataFrame con métricas por modelo
```

---

## 📦 Módulos del Paquete

| Módulo | Funciones Principales |
|---|---|
| `preprocessing` | `split_data()`, `scale_features()`, `encode_categorical()`, `split_timeseries()` |
| `balancing` | `apply_smote()`, `undersample()`, `apply_combined()`, `check_imbalance()` |
| `metrics` | `classification_metrics()`, `regression_metrics()`, `timeseries_metrics()` |
| `validation` | `kfold_validation()`, `stratified_kfold()`, `manual_kfold()` |
| `threshold` | `apply_threshold()`, `optimize_threshold()`, `threshold_analysis()` |
| `eda` | `analisisEDA` — perfilado, limpieza y visualización (CRISP-DM) |
| `models_classification` | `get_classification_models()`, `predict_classification()` |
| `models_regression` | `get_regression_models()` |
| `models_timeseries` | `HoltWintersModel`, `ARIMAModel`, `LSTMModel`, `get_timeseries_models()` |
| `benchmarking` | `run_benchmark()`, `rank_models()`, `benchmark_classification()` |

---

## 🤖 Modelos Implementados

### Clasificación
| Modelo | Notas |
|---|---|
| Logistic Regression | Regularización configurable (L1/L2) |
| Random Forest | 100 árboles por defecto |
| Decision Tree | Criterio Gini / Entropía |
| SVM (RBF) | Probabilidades habilitadas |
| K-Nearest Neighbors | k=5 por defecto |
| Naive Bayes | GaussianNB |
| Gradient Boosting | 100 estimadores por defecto |
| XGBoost *(opcional)* | Si `xgboost` está instalado |

### Regresión
| Modelo | Notas |
|---|---|
| Ridge Regression | α=1.0 por defecto |
| Lasso Regression | α=1.0, max_iter=2000 |
| Random Forest | 100 estimadores |
| Decision Tree | — |
| SVR | Kernel RBF |
| K-Nearest Neighbors | k=5 |
| Gradient Boosting | 100 estimadores |
| XGBoost *(opcional)* | Si `xgboost` está instalado |

### Series de Tiempo
| Modelo | Descripción |
|---|---|
| Holt-Winters | Suavizamiento exponencial triple (additive) |
| Holt-Winters Calibrado | Búsqueda automática de configuración óptima (add/mul) |
| ARIMA(1,1,1) | Orden fijo |
| ARIMA Calibrado | Búsqueda automática de orden (p,d,q) por AIC |
| LSTM | Red recurrente, arquitectura configurable |

---

## 🗄️ Datasets Integrados

| Dataset | Tipo | Muestras | Features | Notas |
|---|---|---|---|---|
| Breast Cancer Wisconsin | Clasificación | 569 | 30 | Ratio clases ≈ 0.59 |
| Credit Card Fraud (Simulado) | Clasificación | 10,000 | 20 | Ratio fraude = 0.02 |
| California Housing | Regresión | 20,640 | 8 | Precio medio vivienda |
| Airline Passengers | Serie de Tiempo | 144 | — | Estacionalidad anual fuerte |

---

## 📊 Funcionalidades de la Interfaz

### 🔍 Exploración & EDA
- KPIs del dataset (filas, columnas, duplicados, nulos)
- Limpieza interactiva: eliminar duplicados, tratar nulos, eliminar columnas
- Estadísticas descriptivas completas (media, mediana, cuantiles, etc.)
- Frecuencia de valores por columna
- Distribución de clases con análisis de desbalanceo
- Visualizaciones EDA: boxplots, histogramas, KDE, correlación, pairplot
- Mapa de correlación interactivo (Plotly)

### ⚙️ Configuración de Modelos
- Selección múltiple de modelos a comparar
- Hiperparámetros ajustables por modelo (con opción "predeterminados")
- Resumen de configuración del experimento

### 🏆 Benchmarking
- Entrenamiento y evaluación automatizada con barra de progreso
- Tabla comparativa con gradiente de color por métricas
- Gráficos de barras comparativos e intervalos de confianza (CV)
- Forecasts vs valores reales (series de tiempo)

### 📈 Resultados Detallados
- Curva ROC con AUC
- Matriz de confusión (heatmap)
- Scores por fold (K-Fold)
- Análisis de threshold: métricas vs umbral de decisión
- Threshold óptimo para F1

### 🥇 Mejor Modelo
- Banner destacado con modelo ganador
- Gráfico radar comparativo (top 5 modelos)
- Recomendaciones basadas en métricas
- Próximos pasos sugeridos

---

## 📐 Métricas de Evaluación

### Clasificación
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- K-Fold CV estratificado (mean ± std)

### Regresión
- MSE, RMSE, MAE, R²
- K-Fold CV (mean ± std)

### Series de Tiempo
- MSE, RMSE, MAE
- MAPE (Mean Absolute Percentage Error)

---

## 🔧 Configuración de Experimentos

| Parámetro | Rango | Predeterminado |
|---|---|---|
| Tamaño del test set | 10 % – 50 % | 30 % |
| Número de folds (K-Fold) | 3 – 10 | 5 |
| Threshold de clasificación | 0.1 – 0.9 | 0.5 |
| Técnica de balanceo | none / smote / undersample / combined | none |
| Escalar features | Sí / No | Sí |
| Train ratio (series de tiempo) | 60 % – 90 % | 80 % |
| Períodos estacionales | 4 / 12 / 24 / 52 | 12 |

---

## 📚 Referencias Principales

> Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

> Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.

> Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

> Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

> Streamlit Inc. (2023). *Streamlit Documentation*. https://docs.streamlit.io

---

<div align="center">

**BCD-7213 · Minería de Datos Avanzada · Universidad LEAD · 2026**  
*Melany Ramírez · Jason Barrantes · Junior Ramírez*

</div>
