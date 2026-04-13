# 🤖 ML Benchmarking System - Caso de Estudio #2

**Sistema de Benchmarking de Modelos de Machine Learning**
**con Web Mining, Redes Neuronales y Reglas de Asociación**

---

**Universidad LEAD**
**BCD-7213 — Minería de Datos Avanzada**
**I Cuatrimestre 2025**

**Profesor:** Dr. Juan Murillo-Morera
**Estudiantes:** Jason Barrantes, Melany Ramírez Anchía y Junior Ramírez Carmona
**Entrega:** Caso de Estudio #2 — 17 de mayo 2025

---

## 📋 Descripción del Proyecto

Este paquete es un **sistema completo de benchmarking** para modelos de Machine Learning con interfaz **Streamlit**.

**Cumple al 100% con los requisitos del Caso de Estudio #2** al integrar en un **único caso práctico** los tres temas solicitados:

* **Web Mining**: Extracción automática de datos desde cualquier sitio web (CSV o scraping de tablas HTML).
* **Redes Neuronales**: Implementación de **exactamente 5 tipos diferentes** de redes neuronales.
* **Reglas de Asociación**: Análisis de patrones de co-ocurrencia mediante el algoritmo Apriori.

**Flujo integrado del caso práctico:**
**Web Mining → Preprocesamiento → Benchmarking con 5 Redes Neuronales → Reglas de Asociación**

---

## 🗂️ Estructura del Proyecto (Caso #2)

```bash
mlbenchmark/
├── mlbenchmark/
│   ├── web_mining.py              ← Web Mining genérico y robusto
│   ├── neural_networks.py         ← 5 tipos de Redes Neuronales
│   ├── association_rules.py       ← Reglas de Asociación (Apriori)
│   ├── preprocessing.py
│   ├── benchmarking.py
│   ├── eda.py
│   └── models_*.py                ← Módulos originales
├── app/
│   └── streamlit_app.py           ← Interfaz Streamlit con soporte Web Mining
└── README.md
```

---

## 🚀 Novedades del Caso de Estudio #2

### 1. Web Mining (`web_mining.py`)

* Soporta cualquier URL (CSV directo o páginas con tablas HTML).
* Scraping genérico con selectores CSS y fallback a datos sintéticos.
* Integrado directamente en la carga de datos del Streamlit.

### 2. Redes Neuronales (`neural_networks.py`)

Se implementaron exactamente 5 tipos diferentes:

* Perceptrón Multicapa (MLP)
* Red Neuronal Recurrente (RNN)
* Red Neuronal Convolucional 1D (CNN)
* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)

Todas compatibles con el sistema de benchmarking y con hiperparámetros optimizables.

### 3. Reglas de Asociación (`association_rules.py`)

* Algoritmo Apriori + métricas completas (support, confidence, lift, conviction).
* Se ejecuta automáticamente al final del flujo sobre los datos extraídos vía Web Mining.

---

## 📥 URLs de Prueba Recomendadas

### CSV directos (rápidos y recomendados)

* Titanic (clasificación): [https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
* Iris (clasificación): [https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
* Wine Quality (clasificación): [https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
* Boston Housing (regresión): [https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv)

### Para probar Web Mining + Scraping

* Books to Scrape (ideal): [http://books.toscrape.com/catalogue/page-1.html](http://books.toscrape.com/catalogue/page-1.html)
* Quotes to Scrape: [http://quotes.toscrape.com/](http://quotes.toscrape.com/)
* Scrapethissite (tabla de países): [https://www.scrapethissite.com/pages/simple/](https://www.scrapethissite.com/pages/simple/)

---

## ⚙️ Instalación y Ejecución

```bash
git clone https://github.com/sklinderton/ML-Benchmarking-System.git
cd ML-Benchmarking-System

python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -e .

streamlit run app/streamlit_app.py
```

---

## 🎯 Cómo Probar el Caso Integrado

1. Ejecuta la aplicación Streamlit.
2. En “Carga de Datos” selecciona “URL del dataset o Web Mining”.
3. Pega cualquiera de las URLs de prueba.
4. Ejecuta el benchmarking completo (incluye las 5 redes neuronales).
5. Al final se generan automáticamente las Reglas de Asociación.

---

**Versión:** 2.0 (Caso de Estudio #2 — Mayo 2025)
**Licencia:** MIT
