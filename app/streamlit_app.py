"""
streamlit_app.py - Aplicación Streamlit para Benchmarking de ML
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
Estudiantes: Melany Ramirez, Jason Barrantes, Junior Ramirez
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mlbenchmark.benchmarking import run_benchmark, rank_models
from mlbenchmark.balancing import check_imbalance
from mlbenchmark.threshold import threshold_analysis, optimize_threshold
from mlbenchmark.metrics import roc_curve_data, precision_recall_curve_data

# ─── Datasets integrados ───────────────────────────────────────────────────────
@st.cache_data
def load_dataset(name, problem_type):
    """Carga datasets predefinidos."""
    from sklearn.datasets import load_breast_cancer, fetch_california_housing
    import pandas as pd
    import numpy as np

    if problem_type == "Clasificación":
        if name == "Breast Cancer Wisconsin":
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name="target")
            return X, y, data.feature_names.tolist()

        elif name == "Credit Card Fraud (Simulado)":
            rng = np.random.RandomState(42)
            n = 10000
            n_fraud = 200
            X_normal = rng.randn(n - n_fraud, 20)
            X_fraud  = rng.randn(n_fraud, 20) + 2.5
            X = np.vstack([X_normal, X_fraud])
            y = np.array([0] * (n - n_fraud) + [1] * n_fraud)
            cols = [f"feature_{i}" for i in range(20)]
            idx = rng.permutation(n)
            return pd.DataFrame(X[idx], columns=cols), pd.Series(y[idx], name="fraud"), cols

    elif problem_type == "Regresión":
        if name == "California Housing":
            data = fetch_california_housing()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name="price")
            return X, y, data.feature_names.tolist()

    elif problem_type == "Series de Tiempo":
        if name == "Airline Passengers":
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
            try:
                df = pd.read_csv(url, header=0, index_col=0, parse_dates=True, squeeze=True)
            except Exception:
                # Datos integrados como fallback
                passengers = [
                    112,118,132,129,121,135,148,148,136,119,104,118,
                    115,126,141,135,125,149,170,170,158,133,114,140,
                    145,150,178,163,172,178,199,199,184,162,146,166,
                    171,180,193,181,183,218,230,242,209,191,172,194,
                    196,196,236,235,229,243,264,272,237,211,180,201,
                    204,188,235,227,234,264,302,293,259,229,203,229,
                    242,233,267,269,270,315,364,347,312,274,237,278,
                    284,277,317,313,318,374,413,405,355,306,271,306,
                    315,301,356,348,355,422,465,467,404,347,305,336,
                    340,318,362,348,363,435,491,505,404,359,310,337,
                    360,342,406,396,420,472,548,559,463,407,362,405,
                    417,391,419,461,472,535,622,606,508,461,390,432,
                ]
                df = pd.Series(passengers, name="passengers")
            return df

    return None


# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Benchmarking System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p { color: #a8b2d8; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .best-model-banner {
        background: linear-gradient(135deg, #0f3460, #e94560);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Encabezado ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 ML Benchmarking System</h1>
    <p>BCD-7213 Minería de Datos Avanzada · Universidad LEAD · I Cuatrimestre 2026</p>
    <p style="color:#e94560; font-size:0.85rem;">Melany Ramírez · Jason Barrantes · Junior Ramírez</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/115px-Tensorflow_logo.svg.png",
             width=60)
    st.title("⚙️ Configuración")
    st.divider()

    # Tipo de problema
    problem_type = st.selectbox(
        "🎯 Tipo de Problema",
        ["Clasificación", "Regresión", "Series de Tiempo"],
        help="Selecciona el tipo de problema de Machine Learning"
    )

    # Dataset según tipo
    st.subheader("📂 Dataset")
    dataset_options = {
        "Clasificación": ["Breast Cancer Wisconsin", "Credit Card Fraud (Simulado)"],
        "Regresión": ["California Housing"],
        "Series de Tiempo": ["Airline Passengers"],
    }
    selected_dataset = st.selectbox("Dataset", dataset_options[problem_type])

    st.divider()

    # Parámetros según tipo
    if problem_type in ["Clasificación", "Regresión"]:
        st.subheader("🔧 Parámetros del Experimento")
        test_size = st.slider("Tamaño del Test Set (%)", 10, 50, 30, 5) / 100
        cv_folds = st.slider("Número de Folds (K-Fold)", 3, 10, 5)
        scale_features_flag = st.checkbox("Escalar Features (StandardScaler)", value=True)

        if problem_type == "Clasificación":
            st.divider()
            st.subheader("📊 Clasificación")
            threshold = st.slider("Threshold de Decisión", 0.1, 0.9, 0.5, 0.05)
            balancing = st.selectbox(
                "Técnica de Balanceo",
                ["none", "smote", "undersample", "combined"],
                format_func=lambda x: {
                    "none": "Sin balanceo",
                    "smote": "SMOTE",
                    "undersample": "Under-sampling",
                    "combined": "Híbrido (SMOTE + Under)",
                }[x]
            )

    elif problem_type == "Series de Tiempo":
        st.subheader("📈 Series de Tiempo")
        train_ratio = st.slider("Ratio de Entrenamiento (%)", 60, 90, 80, 5) / 100
        seasonal_periods = st.selectbox("Períodos Estacionales", [4, 12, 24, 52], index=1)

    st.divider()
    st.caption("💡 Configura los parámetros y carga el dataset para comenzar.")

# ─── Estado de la aplicación ──────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "benchmark_run" not in st.session_state:
    st.session_state.benchmark_run = False
if "results" not in st.session_state:
    st.session_state.results = None

# ─── Tabs principales ─────────────────────────────────────────────────────────
tab_explore, tab_config, tab_bench, tab_detail, tab_best = st.tabs([
    "🔍 Exploración",
    "⚙️ Configuración",
    "🏆 Benchmarking",
    "📊 Resultados Detallados",
    "🥇 Mejor Modelo",
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1: EXPLORACIÓN DE DATOS
# ═════════════════════════════════════════════════════════════════════
with tab_explore:
    st.header("🔍 Exploración del Dataset")

    col_load, col_info = st.columns([1, 3])

    with col_load:
        if st.button("📥 Cargar Dataset", type="primary", use_container_width=True):
            with st.spinner(f"Cargando {selected_dataset}..."):
                try:
                    if problem_type != "Series de Tiempo":
                        X, y, feature_names = load_dataset(selected_dataset, problem_type)
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.feature_names = feature_names
                    else:
                        series = load_dataset(selected_dataset, problem_type)
                        st.session_state.series = series

                    st.session_state.data_loaded = True
                    st.session_state.benchmark_run = False
                    st.success("✅ Dataset cargado!")
                except Exception as e:
                    st.error(f"Error cargando datos: {e}")

    if st.session_state.data_loaded:
        if problem_type != "Series de Tiempo":
            X = st.session_state.X
            y = st.session_state.y

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🗃️ Muestras", f"{len(X):,}")
            col2.metric("📐 Features", len(X.columns))
            col3.metric("🎯 Target", y.name)

            if problem_type == "Clasificación":
                imb = check_imbalance(y.values)
                col4.metric("⚖️ Ratio Clases", f"{imb['ratio']:.3f}")

                # Distribución de clases
                st.subheader("Distribución de Clases")
                c1, c2 = st.columns(2)
                with c1:
                    class_df = pd.DataFrame({
                        "Clase": imb["classes"],
                        "Conteo": imb["counts"],
                    })
                    fig_bar = px.bar(class_df, x="Clase", y="Conteo",
                                     color="Conteo",
                                     color_continuous_scale="RdYlGn",
                                     title="Conteo por Clase")
                    st.plotly_chart(fig_bar, use_container_width=True)

                with c2:
                    fig_pie = px.pie(class_df, values="Conteo", names="Clase",
                                     title="Proporción de Clases",
                                     color_discrete_sequence=["#0f3460","#e94560"])
                    st.plotly_chart(fig_pie, use_container_width=True)

                if imb["is_imbalanced"]:
                    st.warning(f"⚠️ Dataset desbalanceado (ratio={imb['ratio']:.3f}). "
                               f"Severidad: **{imb['severity']}**. Considera usar SMOTE o under-sampling.")

            # Estadísticas descriptivas
            st.subheader("📋 Estadísticas Descriptivas")
            st.dataframe(X.describe().round(3), use_container_width=True)

            # Correlación (top features)
            if len(X.columns) <= 30:
                st.subheader("🔗 Mapa de Correlación")
                corr = X.corr()
                fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
                                     title="Matriz de Correlación",
                                     aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)

        else:
            series = st.session_state.series
            st.metric("📅 Observaciones", len(series))

            fig_ts = px.line(y=series.values if hasattr(series, 'values') else series,
                             title=f"Serie Temporal: {selected_dataset}",
                             labels={"index": "Tiempo", "y": "Valor"})
            fig_ts.update_traces(line_color="#e94560")
            st.plotly_chart(fig_ts, use_container_width=True)

    else:
        st.info("👈 Presiona **Cargar Dataset** para comenzar la exploración.")


# ═════════════════════════════════════════════════════════════════════
# TAB 2: CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════
with tab_config:
    st.header("⚙️ Configuración del Experimento")

    st.info("Los parámetros principales se configuran en el **panel lateral izquierdo**. "
            "Aquí puedes ver y seleccionar los modelos a evaluar.")

    if problem_type in ["Clasificación", "Regresión"]:
        st.subheader("🤖 Modelos Disponibles")

        if problem_type == "Clasificación":
            all_models = [
                "Logistic Regression", "Random Forest", "Decision Tree",
                "SVM", "K-Nearest Neighbors", "Naive Bayes", "Gradient Boosting"
            ]
        else:
            all_models = [
                "Ridge Regression", "Lasso Regression", "Random Forest",
                "Decision Tree", "SVR", "K-Nearest Neighbors", "Gradient Boosting"
            ]

        selected_models = st.multiselect(
            "Selecciona modelos a comparar:",
            all_models,
            default=all_models,
        )
        st.session_state.selected_models = selected_models

    else:
        st.subheader("📈 Modelos de Series de Tiempo")
        ts_models = [
            "Holt-Winters",
            "Holt-Winters Calibrado",
            "ARIMA(1,1,1)",
            "ARIMA Calibrado",
            "LSTM",
        ]
        sel_ts = st.multiselect("Modelos:", ts_models, default=ts_models[:4])
        st.session_state.selected_ts_models = sel_ts

    # Resumen de configuración
    st.divider()
    st.subheader("📋 Resumen de Configuración")
    config_data = {"Parámetro": [], "Valor": []}
    config_data["Parámetro"].append("Tipo de Problema")
    config_data["Valor"].append(problem_type)
    config_data["Parámetro"].append("Dataset")
    config_data["Valor"].append(selected_dataset)

    if problem_type != "Series de Tiempo":
        config_data["Parámetro"].extend(["Test Size", "K-Folds", "Escalar Features"])
        config_data["Valor"].extend([f"{int(test_size*100)}%", cv_folds, scale_features_flag])
        if problem_type == "Clasificación":
            config_data["Parámetro"].extend(["Threshold", "Balanceo"])
            config_data["Valor"].extend([threshold, balancing])
    else:
        config_data["Parámetro"].extend(["Train Ratio", "Períodos Estacionales"])
        config_data["Valor"].extend([f"{int(train_ratio*100)}%", seasonal_periods])

    st.dataframe(pd.DataFrame(config_data), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 3: BENCHMARKING
# ═════════════════════════════════════════════════════════════════════
with tab_bench:
    st.header("🏆 Benchmarking de Modelos")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Primero carga el dataset en la pestaña **Exploración**.")
    else:
        if st.button("🚀 Iniciar Benchmarking", type="primary", use_container_width=True):
            with st.spinner("⏳ Entrenando y evaluando modelos... Esto puede tomar unos momentos."):
                try:
                    if problem_type != "Series de Tiempo":
                        X = st.session_state.X
                        y = st.session_state.y

                        # Filtrar modelos seleccionados
                        from mlbenchmark.models_classification import get_classification_models
                        from mlbenchmark.models_regression import get_regression_models

                        if problem_type == "Clasificación":
                            all_m = get_classification_models()
                            sel = st.session_state.get("selected_models", list(all_m.keys()))
                            models_to_run = {k: v for k, v in all_m.items() if k in sel}
                        else:
                            all_m = get_regression_models()
                            sel = st.session_state.get("selected_models", list(all_m.keys()))
                            models_to_run = {k: v for k, v in all_m.items() if k in sel}

                        pt_map = {
                            "Clasificación": "classification",
                            "Regresión": "regression",
                            "Series de Tiempo": "timeseries",
                        }
                        result = run_benchmark(
                            problem_type=pt_map[problem_type],
                            X=X.values,
                            y=y.values,
                            models=models_to_run,
                            test_size=test_size,
                            cv_folds=cv_folds,
                            threshold=threshold if problem_type == "Clasificación" else 0.5,
                            balancing_technique=balancing if problem_type == "Clasificación" else "none",
                            scale=scale_features_flag,
                        )

                    else:
                        series = st.session_state.series
                        from mlbenchmark.models_timeseries import get_timeseries_models

                        all_ts = get_timeseries_models(seasonal_periods)
                        sel_ts = st.session_state.get("selected_ts_models", list(all_ts.keys()))
                        models_ts = {k: v for k, v in all_ts.items() if k in sel_ts}

                        result = run_benchmark(
                            problem_type="timeseries",
                            series=series.values if hasattr(series, "values") else np.array(series),
                            models=models_ts,
                            seasonal_periods=seasonal_periods,
                            train_ratio=train_ratio,
                        )

                    st.session_state.results = result
                    st.session_state.benchmark_run = True
                    st.success("✅ ¡Benchmarking completado!")

                except Exception as e:
                    import traceback
                    st.error(f"❌ Error durante el benchmarking: {e}")
                    st.code(traceback.format_exc())

        # Mostrar resultados si existen
        if st.session_state.benchmark_run and st.session_state.results:
            res = st.session_state.results
            df = res["results"]
            pt = res["problem_type"]

            st.subheader("📊 Tabla Comparativa de Modelos")

            # Columnas a mostrar (sin columnas internas _)
            display_cols = [c for c in df.columns if not c.startswith("_")]
            display_df = df[display_cols].copy()

            # Color-coding
            st.dataframe(
                display_df.style.format(
                    {c: "{:.4f}" for c in display_df.select_dtypes("float").columns}
                ).background_gradient(
                    subset=[c for c in display_df.columns if c not in ("Model", "CV Scores")],
                    cmap="RdYlGn"
                ),
                use_container_width=True
            )

            # Gráfico de barras comparativo
            st.subheader("📈 Comparación Visual")

            if pt == "classification":
                metric_to_plot = st.selectbox("Métrica a visualizar:",
                                               ["AUC-ROC", "Accuracy", "F1-Score", "Recall", "CV Mean"])
                fig_bar = px.bar(display_df, x="Model", y=metric_to_plot,
                                  color=metric_to_plot,
                                  color_continuous_scale="RdYlGn",
                                  title=f"Comparación por {metric_to_plot}",
                                  text=metric_to_plot)
                fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)

                # Error bars con CV
                if "CV Mean" in display_df.columns and "CV Std" in display_df.columns:
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Bar(
                        x=display_df["Model"],
                        y=display_df["CV Mean"],
                        error_y=dict(type="data", array=display_df["CV Std"]),
                        name="CV Mean ± Std",
                        marker_color="#e94560",
                    ))
                    fig_cv.update_layout(title="K-Fold Cross-Validation (Mean ± Std)",
                                          xaxis_tickangle=-30)
                    st.plotly_chart(fig_cv, use_container_width=True)

            elif pt == "regression":
                c1, c2 = st.columns(2)
                with c1:
                    fig_r2 = px.bar(display_df, x="Model", y="R²",
                                    color="R²", color_continuous_scale="Viridis",
                                    title="R² Score por Modelo", text="R²")
                    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig_r2, use_container_width=True)
                with c2:
                    fig_rmse = px.bar(display_df, x="Model", y="RMSE",
                                      color="RMSE", color_continuous_scale="RdYlGn_r",
                                      title="RMSE por Modelo", text="RMSE")
                    fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig_rmse, use_container_width=True)

            elif pt == "timeseries":
                fig_ts_bar = px.bar(display_df, x="Model", y="RMSE",
                                    color="RMSE", color_continuous_scale="RdYlGn_r",
                                    title="RMSE por Modelo (menor = mejor)", text="RMSE")
                fig_ts_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig_ts_bar, use_container_width=True)

                # Forecasts
                train = res["train"]
                test = res["test"]
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(y=list(train), name="Train",
                                            line=dict(color="#a8b2d8")))
                fig_f.add_trace(go.Scatter(
                    x=list(range(len(train), len(train)+len(test))),
                    y=list(test), name="Real", line=dict(color="#00b4d8", width=2)))

                colors = ["#e94560", "#06d6a0", "#ffd166", "#ef476f", "#118ab2"]
                for i, row in df.iterrows():
                    if row.get("_predictions") is not None:
                        fig_f.add_trace(go.Scatter(
                            x=list(range(len(train), len(train)+len(test))),
                            y=row["_predictions"],
                            name=row["Model"],
                            line=dict(color=colors[i % len(colors)], dash="dash"),
                        ))
                fig_f.update_layout(title="Forecasts vs Valores Reales",
                                     xaxis_title="Tiempo", yaxis_title="Valor")
                st.plotly_chart(fig_f, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 4: RESULTADOS DETALLADOS
# ═════════════════════════════════════════════════════════════════════
with tab_detail:
    st.header("📊 Resultados Detallados por Modelo")

    if not st.session_state.benchmark_run:
        st.warning("⚠️ Ejecuta el benchmarking primero.")
    else:
        res = st.session_state.results
        df = res["results"]
        pt = res["problem_type"]

        if pt == "classification":
            model_names = df["Model"].tolist()
            selected_model = st.selectbox("Selecciona un modelo:", model_names)
            row = df[df["Model"] == selected_model].iloc[0]

            # Métricas principales
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
            c2.metric("Precision", f"{row['Precision']:.4f}")
            c3.metric("Recall", f"{row['Recall']:.4f}")
            c4.metric("F1-Score", f"{row['F1-Score']:.4f}")
            c5.metric("AUC-ROC", f"{row['AUC-ROC']:.4f}")

            col_roc, col_cm = st.columns(2)

            # Curva ROC
            with col_roc:
                y_test = res["y_test"]
                y_prob = row["_y_prob"]
                if y_prob is not None:
                    try:
                        fpr, tpr, _ = roc_curve_data(y_test, y_prob)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy",
                                                      name=f"AUC={row['AUC-ROC']:.4f}",
                                                      line=dict(color="#e94560", width=2)))
                        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                                      line=dict(dash="dash", color="gray"),
                                                      name="Aleatorio"))
                        fig_roc.update_layout(title="Curva ROC",
                                               xaxis_title="FPR (False Positive Rate)",
                                               yaxis_title="TPR (True Positive Rate)")
                        st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo graficar ROC: {e}")

            # Matriz de Confusión
            with col_cm:
                cm = row["_confusion_matrix"]
                if cm:
                    cm_arr = np.array(cm)
                    labels = ["Negativo", "Positivo"]
                    fig_cm = px.imshow(cm_arr, text_auto=True,
                                        x=labels, y=labels,
                                        color_continuous_scale="Blues",
                                        title="Matriz de Confusión",
                                        labels=dict(x="Predicho", y="Real"))
                    st.plotly_chart(fig_cm, use_container_width=True)

            # K-Fold scores
            st.subheader("🔄 Scores por Fold (Cross-Validation)")
            cv_scores = row["CV Scores"]
            if cv_scores:
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                    "AUC-ROC": cv_scores,
                })
                fig_cv = px.bar(fold_df, x="Fold", y="AUC-ROC",
                                 color="AUC-ROC", color_continuous_scale="RdYlGn",
                                 title=f"K-Fold CV | Mean={row['CV Mean']:.4f} ± {row['CV Std']:.4f}")
                fig_cv.add_hline(y=row["CV Mean"], line_dash="dash",
                                  line_color="white", annotation_text="Media")
                st.plotly_chart(fig_cv, use_container_width=True)

            # Análisis de Threshold
            st.subheader("⚖️ Análisis de Threshold")
            y_prob = row["_y_prob"]
            if y_prob is not None:
                thr_data = threshold_analysis(y_test, y_prob)
                thr_df = pd.DataFrame(thr_data)
                fig_thr = go.Figure()
                for col_name in ["accuracy", "precision", "recall", "f1"]:
                    fig_thr.add_trace(go.Scatter(
                        x=thr_df["threshold"], y=thr_df[col_name],
                        name=col_name.capitalize(), mode="lines"
                    ))
                fig_thr.update_layout(title="Métricas vs Threshold",
                                       xaxis_title="Threshold",
                                       yaxis_title="Score")
                st.plotly_chart(fig_thr, use_container_width=True)

                # Threshold óptimo
                opt = optimize_threshold(y_test, y_prob, metric="f1")
                st.info(f"🎯 Threshold óptimo para F1: **{opt['optimal_threshold']}** "
                        f"(F1={opt['best_score']:.4f})")

        elif pt == "regression":
            model_names = df["Model"].tolist()
            sel = st.selectbox("Selecciona un modelo:", model_names)
            row = df[df["Model"] == sel].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{row['R²']:.4f}")
            c2.metric("RMSE", f"{row['RMSE']:.4f}")
            c3.metric("MAE", f"{row['MAE']:.4f}")
            c4.metric("CV Mean (R²)", f"{row['CV Mean (R²)']:.4f} ± {row['CV Std']:.4f}")

            cv_scores = row["CV Scores"]
            if cv_scores:
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                    "R²": cv_scores,
                })
                fig_cv = px.bar(fold_df, x="Fold", y="R²",
                                 color="R²", color_continuous_scale="RdYlGn",
                                 title=f"K-Fold CV | Mean={row['CV Mean (R²)']:.4f}")
                st.plotly_chart(fig_cv, use_container_width=True)

        elif pt == "timeseries":
            model_names = df["Model"].tolist()
            sel = st.selectbox("Selecciona un modelo:", model_names)
            row = df[df["Model"] == sel].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{row['RMSE']:.4f}" if row["RMSE"] else "N/A")
            c2.metric("MAE", f"{row['MAE']:.4f}" if row["MAE"] else "N/A")
            c3.metric("MSE", f"{row['MSE']:.4f}" if row["MSE"] else "N/A")
            c4.metric("MAPE", f"{row['MAPE (%)']:.2f}%" if row["MAPE (%)"] else "N/A")

            if row["_predictions"]:
                train = res["train"]
                test = res["test"]
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(y=list(train), name="Entrenamiento",
                                            line=dict(color="#a8b2d8")))
                fig_f.add_trace(go.Scatter(
                    x=list(range(len(train), len(train)+len(test))),
                    y=list(test), name="Real", line=dict(color="#00b4d8", width=2)))
                fig_f.add_trace(go.Scatter(
                    x=list(range(len(train), len(train)+len(test))),
                    y=row["_predictions"], name="Predicción",
                    line=dict(color="#e94560", dash="dash", width=2)))
                fig_f.update_layout(title=f"Forecast: {sel}",
                                     xaxis_title="Tiempo", yaxis_title="Valor")
                st.plotly_chart(fig_f, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 5: MEJOR MODELO
# ═════════════════════════════════════════════════════════════════════
with tab_best:
    st.header("🥇 Mejor Modelo")

    if not st.session_state.benchmark_run:
        st.warning("⚠️ Ejecuta el benchmarking primero.")
    else:
        res = st.session_state.results
        df = res["results"]
        pt = res["problem_type"]

        # Determinar mejor modelo
        best = df.iloc[0]

        # Banner
        if pt == "classification":
            primary_metric = "AUC-ROC"
            primary_value = best["AUC-ROC"]
        elif pt == "regression":
            primary_metric = "R²"
            primary_value = best["R²"]
        else:
            primary_metric = "RMSE"
            primary_value = best["RMSE"]

        st.markdown(f"""
        <div class="best-model-banner">
            <h2>🏆 {best['Model']}</h2>
            <h3>{primary_metric}: {primary_value:.4f}</h3>
            <p>Mejor modelo según la métrica principal</p>
        </div>
        """, unsafe_allow_html=True)

        # Detalles del ganador
        st.subheader("📋 Métricas del Mejor Modelo")
        display_cols = [c for c in df.columns if not c.startswith("_")]
        best_display = df[display_cols].iloc[0:1]
        st.dataframe(best_display.style.format(
            {c: "{:.4f}" for c in best_display.select_dtypes("float").columns}
        ), use_container_width=True)

        # Radar chart comparativo (clasificación)
        if pt == "classification":
            st.subheader("🕸️ Comparación Radar")
            metrics_radar = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
            display_df = df[[c for c in df.columns if not c.startswith("_")]].copy()
            top_n = min(5, len(display_df))

            fig_radar = go.Figure()
            colors_r = ["#e94560", "#06d6a0", "#ffd166", "#118ab2", "#9b5de5"]
            for i, row in display_df.head(top_n).iterrows():
                values = [row[m] for m in metrics_radar]
                values.append(values[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_radar + [metrics_radar[0]],
                    name=row["Model"],
                    line=dict(color=colors_r[i % len(colors_r)]),
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0.0, 1.0])),
                                     title="Comparación Multimétrica (Top 5 Modelos)")
            st.plotly_chart(fig_radar, use_container_width=True)

        # Recomendaciones
        st.subheader("💡 Recomendaciones")
        st.success(f"✅ Se recomienda usar **{best['Model']}** para este problema.")

        if pt == "classification":
            if primary_value >= 0.95:
                st.info("🌟 Rendimiento excelente. El modelo es altamente discriminativo.")
            elif primary_value >= 0.85:
                st.info("👍 Buen rendimiento. Considera optimizar hiperparámetros para mejorar.")
            else:
                st.warning("⚠️ Rendimiento moderado. Considera más datos o feature engineering.")

        elif pt == "regression":
            if primary_value >= 0.85:
                st.info("🌟 El modelo explica más del 85% de la varianza. Excelente ajuste.")
            elif primary_value >= 0.70:
                st.info("👍 Buen ajuste. Prueba con más features o transformaciones.")
            else:
                st.warning("⚠️ R² bajo. El modelo puede estar subajustando (underfitting).")

        elif pt == "timeseries":
            mape = best.get("MAPE (%)")
            if mape and mape < 5:
                st.info("🌟 MAPE < 5%: Forecasts muy precisos.")
            elif mape and mape < 10:
                st.info("👍 MAPE < 10%: Forecasts aceptables.")
            else:
                st.warning("⚠️ MAPE alto. Considera más datos o ajustar períodos estacionales.")

        # Próximos pasos
        st.subheader("🚀 Próximos Pasos Sugeridos")
        st.markdown("""
        1. **Optimización de hiperparámetros**: Usa Grid Search o Random Search con Optuna.
        2. **Interpretabilidad**: Analiza SHAP values y feature importance.
        3. **Validación adicional**: Prueba con datos externos para confirmar generalización.
        4. **Monitoreo**: Implementa detección de model drift en producción.
        5. **AutoML**: Considera bibliotecas como AutoSklearn o H2O.ai para automatizar.
        """)