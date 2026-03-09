"""
streamlit_app.py - Aplicación Streamlit para Benchmarking de ML
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
Estudiantes: Melany Ramirez, Jason Barrantes, Junior Ramirez
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mlbenchmark.benchmarking import run_benchmark
from mlbenchmark.balancing import check_imbalance
from mlbenchmark.metrics import roc_curve_data
from mlbenchmark.threshold import optimize_threshold, threshold_analysis


# =========================
# CONFIG VISUAL
# =========================
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SCALE_MAIN = "Viridis"
COLOR_SCALE_REVERSE = "Plasma"
DISCRETE_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]


# =========================
# HELPERS
# =========================
def style_color_table(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    styler = df.style

    float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns.tolist()
    if float_cols:
        styler = styler.format({c: "{:.4f}" for c in float_cols})

    if num_cols:
        styler = styler.background_gradient(subset=num_cols, cmap="viridis")
        styler = styler.set_properties(subset=num_cols, **{"color": "white"})

    styler = styler.set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#1e1e1e"),
                ("color", "#e0e0e0"),
                ("border-color", "#333"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("border-color", "#333"),
            ],
        },
    ])
    return styler


@st.cache_data
def load_dataset(name, problem_type):
    """Carga datasets predefinidos."""
    from sklearn.datasets import fetch_california_housing, load_breast_cancer

    if problem_type == "Clasificación":
        if name == "Breast Cancer Wisconsin":
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name="target")
            return X, y, data.feature_names.tolist()

        if name == "Credit Card Fraud (Simulado)":
            rng = np.random.RandomState(42)
            n = 10000
            n_fraud = 200
            X_normal = rng.randn(n - n_fraud, 20)
            X_fraud = rng.randn(n_fraud, 20) + 2.5
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
                df = pd.read_csv(url, header=0, index_col=0, parse_dates=True).squeeze("columns")
            except Exception:
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


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Lee un archivo subido (csv/xlsx/json) en un DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no soportado. Usa CSV, Excel o JSON.")


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ML Benchmarking System - Junior Ramirez, Jason Barrantes, Melany Ramirez",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap');

    html, body, [class*="css"], [class*="st-"], .stApp, .stApp * {
        font-family: 'Poppins', sans-serif !important;
    }

    .stApp {
        background: #121212;
        color: #e0e0e0;
    }

    section[data-testid="stSidebar"] {
        background: #1e1e1e;
        border-right: 1px solid #333;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #333;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    p, li, span, label, div {
        color: #d6d6d6;
    }

    .main-header {
        background: linear-gradient(135deg, #121212 0%, #202020 50%, #1a1a1a 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #333;
        box-shadow: 0 8px 18px rgba(0,0,0,0.35);
    }
    .main-header h1 {
        color: #e0e0e0;
        font-size: 2.2rem;
        margin: 0;
    }
    .main-header p {
        color: #bdbdbd;
        margin: 0.5rem 0 0 0;
    }

    .best-model-banner {
        background: linear-gradient(135deg, #1e1e1e 0%, #333 60%, #2a2a2a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #444;
        box-shadow: 0 10px 22px rgba(0,0,0,0.35);
    }
    .best-model-banner h2, .best-model-banner h3, .best-model-banner p {
        color: #e0e0e0;
        margin: 0.25rem 0;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select, .stMultiSelect div, .stTextArea textarea {
        background: #1e1e1e !important;
        color: #e0e0e0 !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div {
        background: #1e1e1e !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
        color: #e0e0e0 !important;
    }
    .stSlider [data-baseweb="slider"] * {
        color: #e0e0e0 !important;
    }

    .stButton button {
        background: #333 !important;
        color: #e0e0e0 !important;
        border: 1px solid #555 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.55rem 0.9rem !important;
    }
    .stButton button:hover {
        background: #3a3a3a !important;
        border-color: #777 !important;
    }

    div[data-testid="stDataFrame"] {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 0.25rem;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 10px;
        border: 1px solid #444;
        background: #1e1e1e;
        color: #e0e0e0;
    }

    hr {
        border-color: #333 !important;
    }
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
# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("Configuración")
    st.divider()

    problem_type = st.selectbox(
        "Tipo de Problema",
        ["Clasificación", "Regresión", "Series de Tiempo"],
        help="Selecciona el tipo de problema de Machine Learning",
        key="sidebar_problem_type",
    )

    st.subheader("Dataset")
    dataset_options = {
        "Clasificación": ["Breast Cancer Wisconsin", "Credit Card Fraud (Simulado)", "Archivo (CSV/Excel/JSON)"],
        "Regresión": ["California Housing", "Archivo (CSV/Excel/JSON)"],
        "Series de Tiempo": ["Airline Passengers", "Archivo (CSV/Excel/JSON)"],
    }
    selected_dataset = st.selectbox(
        "Dataset",
        dataset_options[problem_type],
        key="sidebar_dataset",
    )

    uploaded_file = None
    if selected_dataset == "Archivo (CSV/Excel/JSON)":
        uploaded_file = st.file_uploader(
            "Sube tu dataset",
            type=["csv", "xlsx", "xls", "json"],
            accept_multiple_files=False,
            key="sidebar_file_uploader",
        )

    st.divider()

    threshold = 0.5
    balancing = "none"
    train_ratio = 0.8
    seasonal_periods = 12

    if problem_type in ["Clasificación", "Regresión"]:
        st.subheader("Parámetros del Experimento")
        test_size = st.slider(
            "Tamaño del Test Set (%)",
            10,
            50,
            30,
            5,
            key="sidebar_test_size_slider",
        ) / 100

        cv_folds = st.slider(
            "Número de Folds (K-Fold)",
            3,
            10,
            5,
            key="sidebar_cv_folds_slider",
        )

        scale_features_flag = st.checkbox(
            "Escalar Features (StandardScaler)",
            value=True,
            key="sidebar_scale_features_checkbox",
        )

        if problem_type == "Clasificación":
            st.divider()
            st.subheader("Clasificación")

            threshold = st.slider(
                "Threshold de Decisión",
                0.1,
                0.9,
                0.5,
                0.05,
                key="sidebar_threshold_slider",
            )

            balancing = st.selectbox(
                "Técnica de Balanceo",
                ["none", "smote", "undersample", "combined"],
                format_func=lambda x: {
                    "none": "Sin balanceo",
                    "smote": "SMOTE",
                    "undersample": "Under-sampling",
                    "combined": "Híbrido (SMOTE + Under)",
                }[x],
                key="sidebar_balancing_selectbox",
            )
    else:
        scale_features_flag = True
        st.subheader("Series de Tiempo")
        train_ratio = st.slider(
            "Ratio de Entrenamiento (%)",
            60,
            90,
            80,
            5,
            key="sidebar_train_ratio_slider",
        ) / 100

        seasonal_periods = st.selectbox(
            "Períodos Estacionales",
            [4, 12, 24, 52],
            index=1,
            key="sidebar_seasonal_periods_selectbox",
        )

    st.divider()
    st.caption("Configura los parámetros y carga el dataset para comenzar.")


# =========================
# SESSION STATE
# =========================
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "benchmark_run" not in st.session_state:
    st.session_state.benchmark_run = False
if "results" not in st.session_state:
    st.session_state.results = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None


# =========================
# TABS
# =========================
tab_explore, tab_config, tab_bench, tab_detail, tab_best = st.tabs([
    "Exploración de Datos",
    "Configuración del Experimento",
    "Benchmarking de Modelos",
    "Resultados Detallados",
    "Mejor Modelo",
])


# =========================
# TAB 1: EXPLORACIÓN
# =========================
with tab_explore:
    st.header("Exploración del Dataset")

    col_load, col_info = st.columns([1, 3])

    with col_load:
        if st.button("Cargar Dataset", type="primary", use_container_width=True, key="btn_load_dataset"):
            with st.spinner(f"Cargando {selected_dataset}..."):
                try:
                    if selected_dataset == "Archivo (CSV/Excel/JSON)":
                        if uploaded_file is None:
                            st.error("Sube un archivo primero.")
                            st.stop()

                        dfu = read_uploaded_table(uploaded_file)
                        dfu.columns = [str(c).strip() for c in dfu.columns]
                        st.session_state.uploaded_df = dfu
                        st.session_state.data_loaded = True
                        st.session_state.benchmark_run = False
                        st.session_state.results = None
                        st.success("Archivo cargado. Configura las columnas abajo.")
                        st.stop()

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
                    st.session_state.results = None
                    st.success("Dataset cargado.")
                except Exception as e:
                    st.error(f"Error cargando datos: {e}")

    if st.session_state.data_loaded:
        if selected_dataset == "Archivo (CSV/Excel/JSON)" and st.session_state.uploaded_df is not None:
            dfu = st.session_state.uploaded_df

            st.subheader("Configurar dataset (archivo)")

            if problem_type in ["Clasificación", "Regresión"]:
                target_col = st.selectbox(
                    "Columna target",
                    dfu.columns.tolist(),
                    key="explore_target_col_selectbox",
                )

                X_raw = dfu.drop(columns=[target_col]).copy()
                y_raw = dfu[target_col].copy()
                X_enc = pd.get_dummies(X_raw, drop_first=False)

                if problem_type == "Regresión":
                    y_num = pd.to_numeric(y_raw, errors="coerce")
                    if y_num.isna().any():
                        st.error("La columna target para regresión debe ser numérica.")
                        st.stop()
                    y_final = pd.Series(y_num.values, name=str(target_col))
                else:
                    if y_raw.dtype == "object" or str(y_raw.dtype).startswith("category"):
                        y_codes, _ = pd.factorize(y_raw.astype(str))
                        y_final = pd.Series(y_codes, name=str(target_col))
                    else:
                        y_final = pd.Series(y_raw.values, name=str(target_col))

                st.session_state.X = X_enc
                st.session_state.y = y_final
                st.session_state.feature_names = X_enc.columns.tolist()
                st.success("Dataset listo para benchmarking.")

            else:
                date_col = st.selectbox(
                    "Columna de fecha",
                    dfu.columns.tolist(),
                    key="explore_date_col_selectbox",
                )
                value_options = [c for c in dfu.columns if c != date_col]
                value_col = st.selectbox(
                    "Columna de valor",
                    value_options,
                    key="explore_value_col_selectbox",
                )

                df_ts = dfu[[date_col, value_col]].copy()
                df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors="coerce")
                df_ts = df_ts.dropna(subset=[date_col])
                df_ts = df_ts.sort_values(date_col)

                series = pd.Series(df_ts[value_col].values, index=df_ts[date_col].values, name=str(value_col))
                st.session_state.series = series
                st.success("Serie temporal lista para benchmarking.")

        if problem_type != "Series de Tiempo":
            X = st.session_state.X
            y = st.session_state.y

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Muestras", f"{len(X):,}")
            col2.metric("Features", len(X.columns))
            col3.metric("Target", y.name)

            if problem_type == "Clasificación":
                imb = check_imbalance(y.values)
                col4.metric("Ratio de Clases", f"{imb['ratio']:.3f}")

                st.subheader("Distribución de Clases")
                c1, c2 = st.columns(2)

                with c1:
                    class_df = pd.DataFrame({
                        "Clase": imb["classes"],
                        "Conteo": imb["counts"],
                    })
                    fig_bar = px.bar(
                        class_df,
                        x="Clase",
                        y="Conteo",
                        color="Conteo",
                        color_continuous_scale=COLOR_SCALE_MAIN,
                        title="Conteo por Clase",
                        template=PLOTLY_TEMPLATE,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with c2:
                    fig_pie = px.pie(
                        class_df,
                        values="Conteo",
                        names="Clase",
                        title="Proporción de Clases",
                        color_discrete_sequence=DISCRETE_COLORS,
                        template=PLOTLY_TEMPLATE,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                if imb["is_imbalanced"]:
                    st.warning(
                        f"Dataset desbalanceado (ratio={imb['ratio']:.3f}). "
                        f"Severidad: **{imb['severity']}**. Considera usar SMOTE o under-sampling."
                    )

            st.subheader("Estadísticas Descriptivas")
            st.dataframe(X.describe().round(3), use_container_width=True)

            if len(X.columns) <= 30:
                st.subheader("Mapa de Correlación")
                corr = X.corr(numeric_only=True)
                fig_corr = px.imshow(
                    corr,
                    color_continuous_scale=COLOR_SCALE_MAIN,
                    title="Matriz de Correlación",
                    aspect="auto",
                )
                fig_corr.update_layout(template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_corr, use_container_width=True)

        else:
            series = st.session_state.series
            st.metric("Observaciones", len(series))

            fig_ts = px.line(
                y=series.values if hasattr(series, "values") else series,
                title=f"Serie Temporal: {selected_dataset}",
                labels={"index": "Tiempo", "y": "Valor"},
                template=PLOTLY_TEMPLATE,
            )
            fig_ts.update_traces(line_color=DISCRETE_COLORS[0])
            st.plotly_chart(fig_ts, use_container_width=True)

    else:
        st.info("Presiona Cargar Dataset para comenzar la exploración.")


# =========================
# TAB 2: CONFIG
# =========================
with tab_config:
    st.header("Configuración del Experimento")

    st.info(
        "Los parámetros principales se configuran en el panel lateral izquierdo. "
        "Aquí puedes ver y seleccionar los modelos a evaluar."
    )

    if problem_type in ["Clasificación", "Regresión"]:
        st.subheader("Modelos Disponibles")

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
            key="config_models_multiselect",
        )
        st.session_state.selected_models = selected_models

    else:
        st.subheader("Modelos de Series de Tiempo")
        ts_models = [
            "Holt-Winters",
            "Holt-Winters Calibrado",
            "ARIMA(1,1,1)",
            "ARIMA Calibrado",
            "LSTM",
        ]
        sel_ts = st.multiselect(
            "Modelos:",
            ts_models,
            default=ts_models[:4],
            key="config_ts_models_multiselect",
        )
        st.session_state.selected_ts_models = sel_ts

    st.divider()
    st.subheader("Resumen de Configuración")

    config_data = {"Parámetro": [], "Valor": []}
    config_data["Parámetro"].append("Tipo de Problema")
    config_data["Valor"].append(problem_type)
    config_data["Parámetro"].append("Dataset")
    config_data["Valor"].append(selected_dataset)

    if problem_type != "Series de Tiempo":
        config_data["Parámetro"].extend(["Test Size", "K-Folds", "Escalar Features"])
        config_data["Valor"].extend([
            f"{int(test_size * 100)}%",
            cv_folds,
            scale_features_flag,
        ])

        if problem_type == "Clasificación":
            config_data["Parámetro"].extend(["Threshold", "Balanceo"])
            config_data["Valor"].extend([threshold, balancing])

    else:
        config_data["Parámetro"].extend(["Train Ratio", "Períodos Estacionales"])
        config_data["Valor"].extend([
            f"{int(train_ratio * 100)}%",
            seasonal_periods,
        ])

    config_df = pd.DataFrame(config_data)
    config_df["Valor"] = config_df["Valor"].astype(str)
    st.dataframe(config_df, use_container_width=True)


# =========================
# TAB 3: BENCHMARKING
# =========================
with tab_bench:
    st.header("Benchmarking de Modelos")

    if not st.session_state.data_loaded:
        st.warning("Primero carga el dataset en la pestaña Exploración de Datos.")
    else:
        if selected_dataset == "Archivo (CSV/Excel/JSON)":
            if problem_type in ["Clasificación", "Regresión"] and (
                "X" not in st.session_state or "y" not in st.session_state
            ):
                st.warning("Configura la columna target en Exploración de Datos antes de iniciar.")
            if problem_type == "Series de Tiempo" and "series" not in st.session_state:
                st.warning("Configura las columnas de fecha y valor en Exploración de Datos antes de iniciar.")

        if st.button("Iniciar Benchmarking", type="primary", use_container_width=True, key="btn_run_benchmark"):
            with st.spinner("Entrenando y evaluando modelos..."):
                try:
                    if problem_type != "Series de Tiempo":
                        X = st.session_state.X
                        y = st.session_state.y

                        from mlbenchmark.models_classification import get_classification_models
                        from mlbenchmark.models_regression import get_regression_models

                        if problem_type == "Clasificación":
                            all_m = get_classification_models()
                            sel = st.session_state.get("selected_models", list(all_m.keys()))
                            models_to_run = {k: v for k, v in all_m.items() if k in sel}
                            pt_key = "classification"
                            benchmark_threshold = threshold
                            benchmark_balancing = balancing
                        else:
                            all_m = get_regression_models()
                            sel = st.session_state.get("selected_models", list(all_m.keys()))
                            models_to_run = {k: v for k, v in all_m.items() if k in sel}
                            pt_key = "regression"
                            benchmark_threshold = 0.5
                            benchmark_balancing = "none"

                        result = run_benchmark(
                            problem_type=pt_key,
                            X=X.values,
                            y=y.values,
                            models=models_to_run,
                            test_size=test_size,
                            cv_folds=cv_folds,
                            threshold=benchmark_threshold,
                            balancing_technique=benchmark_balancing,
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
                    st.success("Benchmarking completado.")

                except Exception as e:
                    import traceback
                    st.error(f"Error durante el benchmarking: {e}")
                    st.code(traceback.format_exc())

        if st.session_state.benchmark_run and st.session_state.results:
            res = st.session_state.results
            df = res["results"]
            pt = res["problem_type"]

            st.subheader("Tabla Comparativa de Modelos")

            display_cols = [c for c in df.columns if not c.startswith("_")]
            display_df = df[display_cols].copy()

            st.dataframe(
                style_color_table(display_df),
                use_container_width=True,
            )

            st.subheader("Comparación Visual")

            if pt == "classification":
                metric_to_plot = st.selectbox(
                    "Métrica a visualizar:",
                    ["AUC-ROC", "Accuracy", "F1-Score", "Recall", "CV Mean"],
                    key="bench_metric_to_plot_selectbox",
                )

                fig_bar = px.bar(
                    display_df,
                    x="Model",
                    y=metric_to_plot,
                    color=metric_to_plot,
                    color_continuous_scale=COLOR_SCALE_MAIN,
                    title=f"Comparación por {metric_to_plot}",
                    text=metric_to_plot,
                    template=PLOTLY_TEMPLATE,
                )
                fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)

                if "CV Mean" in display_df.columns and "CV Std" in display_df.columns:
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Bar(
                        x=display_df["Model"],
                        y=display_df["CV Mean"],
                        error_y=dict(type="data", array=display_df["CV Std"]),
                        name="CV Mean ± Std",
                        marker_color=DISCRETE_COLORS[0],
                    ))
                    fig_cv.update_layout(
                        title="K-Fold Cross-Validation (Mean ± Std)",
                        xaxis_tickangle=-30,
                        template=PLOTLY_TEMPLATE,
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)

            elif pt == "regression":
                c1, c2 = st.columns(2)

                with c1:
                    fig_r2 = px.bar(
                        display_df,
                        x="Model",
                        y="R²",
                        color="R²",
                        color_continuous_scale=COLOR_SCALE_MAIN,
                        title="R² Score por Modelo",
                        text="R²",
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig_r2, use_container_width=True)

                with c2:
                    fig_rmse = px.bar(
                        display_df,
                        x="Model",
                        y="RMSE",
                        color="RMSE",
                        color_continuous_scale=COLOR_SCALE_REVERSE,
                        title="RMSE por Modelo",
                        text="RMSE",
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig_rmse, use_container_width=True)

            elif pt == "timeseries":
                fig_ts_bar = px.bar(
                    display_df,
                    x="Model",
                    y="RMSE",
                    color="RMSE",
                    color_continuous_scale=COLOR_SCALE_REVERSE,
                    title="RMSE por Modelo (menor = mejor)",
                    text="RMSE",
                    template=PLOTLY_TEMPLATE,
                )
                fig_ts_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig_ts_bar, use_container_width=True)

                train = res["train"]
                test = res["test"]

                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    y=list(train),
                    name="Train",
                    line=dict(color=DISCRETE_COLORS[8]),
                ))
                fig_f.add_trace(go.Scatter(
                    x=list(range(len(train), len(train) + len(test))),
                    y=list(test),
                    name="Real",
                    line=dict(color=DISCRETE_COLORS[2], width=2),
                ))

                colors = DISCRETE_COLORS
                for i, row in df.iterrows():
                    if row.get("_predictions") is not None:
                        fig_f.add_trace(go.Scatter(
                            x=list(range(len(train), len(train) + len(test))),
                            y=row["_predictions"],
                            name=row["Model"],
                            line=dict(color=colors[i % len(colors)], dash="dash"),
                        ))

                fig_f.update_layout(
                    title="Forecasts vs Valores Reales",
                    xaxis_title="Tiempo",
                    yaxis_title="Valor",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_f, use_container_width=True)


# =========================
# TAB 4: DETALLE
# =========================
with tab_detail:
    st.header("Resultados Detallados por Modelo")

    if not st.session_state.benchmark_run:
        st.warning("Ejecuta el benchmarking primero.")
    else:
        res = st.session_state.results
        df = res["results"]
        pt = res["problem_type"]

        if pt == "classification":
            model_names = df["Model"].tolist()
            selected_model = st.selectbox(
                "Selecciona un modelo:",
                model_names,
                key="detail_selected_classification_model",
            )
            row = df[df["Model"] == selected_model].iloc[0]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
            c2.metric("Precision", f"{row['Precision']:.4f}")
            c3.metric("Recall", f"{row['Recall']:.4f}")
            c4.metric("F1-Score", f"{row['F1-Score']:.4f}")
            c5.metric("AUC-ROC", f"{row['AUC-ROC']:.4f}")

            col_roc, col_cm = st.columns(2)

            with col_roc:
                y_test = res["y_test"]
                y_prob = row["_y_prob"]
                if y_prob is not None:
                    try:
                        fpr, tpr, _ = roc_curve_data(y_test, y_prob)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr,
                            y=tpr,
                            fill="tozeroy",
                            name=f"AUC={row['AUC-ROC']:.4f}",
                            line=dict(color=DISCRETE_COLORS[0], width=2),
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            line=dict(dash="dash", color=DISCRETE_COLORS[9]),
                            name="Aleatorio",
                        ))
                        fig_roc.update_layout(
                            title="Curva ROC",
                            xaxis_title="FPR (False Positive Rate)",
                            yaxis_title="TPR (True Positive Rate)",
                            template=PLOTLY_TEMPLATE,
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo graficar ROC: {e}")

            with col_cm:
                cm = row["_confusion_matrix"]
                if cm is not None:
                    cm_arr = np.array(cm)
                    labels = ["Negativo", "Positivo"]
                    fig_cm = px.imshow(
                        cm_arr,
                        text_auto=True,
                        x=labels,
                        y=labels,
                        color_continuous_scale=COLOR_SCALE_MAIN,
                        title="Matriz de Confusión",
                        labels=dict(x="Predicho", y="Real"),
                    )
                    fig_cm.update_layout(template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_cm, use_container_width=True)

            st.subheader("Scores por Fold (Cross-Validation)")
            cv_scores = row["CV Scores"]
            if cv_scores is not None and len(cv_scores) > 0:
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                    "AUC-ROC": cv_scores,
                })
                fig_cv = px.bar(
                    fold_df,
                    x="Fold",
                    y="AUC-ROC",
                    color="AUC-ROC",
                    color_continuous_scale=COLOR_SCALE_MAIN,
                    title=f"K-Fold CV | Mean={row['CV Mean']:.4f} ± {row['CV Std']:.4f}",
                    template=PLOTLY_TEMPLATE,
                )
                fig_cv.add_hline(
                    y=row["CV Mean"],
                    line_dash="dash",
                    line_color="white",
                    annotation_text="Media",
                )
                st.plotly_chart(fig_cv, use_container_width=True)

            st.subheader("Análisis de Threshold")
            if y_prob is not None:
                thr_data = threshold_analysis(y_test, y_prob)
                thr_df = pd.DataFrame(thr_data)
                fig_thr = go.Figure()
                line_colors = [DISCRETE_COLORS[0], DISCRETE_COLORS[1], DISCRETE_COLORS[2], DISCRETE_COLORS[3]]

                for i, col_name in enumerate(["accuracy", "precision", "recall", "f1"]):
                    fig_thr.add_trace(go.Scatter(
                        x=thr_df["threshold"],
                        y=thr_df[col_name],
                        name=col_name.capitalize(),
                        mode="lines",
                        line=dict(color=line_colors[i]),
                    ))

                fig_thr.update_layout(
                    title="Métricas vs Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="Score",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_thr, use_container_width=True)

                opt = optimize_threshold(y_test, y_prob, metric="f1")
                st.info(
                    f"Threshold óptimo para F1: **{opt['optimal_threshold']}** "
                    f"(F1={opt['best_score']:.4f})"
                )

        elif pt == "regression":
            model_names = df["Model"].tolist()
            sel = st.selectbox(
                "Selecciona un modelo:",
                model_names,
                key="detail_selected_regression_model",
            )
            row = df[df["Model"] == sel].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{row['R²']:.4f}")
            c2.metric("RMSE", f"{row['RMSE']:.4f}")
            c3.metric("MAE", f"{row['MAE']:.4f}")
            c4.metric("CV Mean (R²)", f"{row['CV Mean (R²)']:.4f} ± {row['CV Std']:.4f}")

            cv_scores = row["CV Scores"]
            if cv_scores is not None and len(cv_scores) > 0:
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                    "R²": cv_scores,
                })
                fig_cv = px.bar(
                    fold_df,
                    x="Fold",
                    y="R²",
                    color="R²",
                    color_continuous_scale=COLOR_SCALE_MAIN,
                    title=f"K-Fold CV | Mean={row['CV Mean (R²)']:.4f}",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_cv, use_container_width=True)

        elif pt == "timeseries":
            model_names = df["Model"].tolist()
            sel = st.selectbox(
                "Selecciona un modelo:",
                model_names,
                key="detail_selected_timeseries_model",
            )
            row = df[df["Model"] == sel].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{row['RMSE']:.4f}" if pd.notna(row["RMSE"]) else "N/A")
            c2.metric("MAE", f"{row['MAE']:.4f}" if pd.notna(row["MAE"]) else "N/A")
            c3.metric("MSE", f"{row['MSE']:.4f}" if pd.notna(row["MSE"]) else "N/A")
            c4.metric("MAPE", f"{row['MAPE (%)']:.2f}%" if pd.notna(row["MAPE (%)"]) else "N/A")

            if row.get("_predictions") is not None and len(row["_predictions"]) > 0:
                train = res["train"]
                test = res["test"]

                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    y=list(train),
                    name="Entrenamiento",
                    line=dict(color=DISCRETE_COLORS[8]),
                ))
                fig_f.add_trace(go.Scatter(
                    x=list(range(len(train), len(train) + len(test))),
                    y=list(test),
                    name="Real",
                    line=dict(color=DISCRETE_COLORS[2], width=2),
                ))
                fig_f.add_trace(go.Scatter(
                    x=list(range(len(train), len(train) + len(test))),
                    y=row["_predictions"],
                    name="Predicción",
                    line=dict(color=DISCRETE_COLORS[0], dash="dash", width=2),
                ))
                fig_f.update_layout(
                    title=f"Forecast: {sel}",
                    xaxis_title="Tiempo",
                    yaxis_title="Valor",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_f, use_container_width=True)


# =========================
# TAB 5: MEJOR MODELO
# =========================
with tab_best:
    st.header("Mejor Modelo")

    if not st.session_state.benchmark_run:
        st.warning("Ejecuta el benchmarking primero.")
    else:
        res = st.session_state.results
        df = res["results"]
        pt = res["problem_type"]

        best = df.iloc[0]

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
            <h2>{best['Model']}</h2>
            <h3>{primary_metric}: {primary_value:.4f}</h3>
            <p>Mejor modelo según la métrica principal</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Métricas del Mejor Modelo")
        display_cols = [c for c in df.columns if not c.startswith("_")]
        best_display = df[display_cols].iloc[0:1]
        st.dataframe(style_color_table(best_display), use_container_width=True)

        if pt == "classification":
            st.subheader("Comparación Radar")
            metrics_radar = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
            display_df = df[[c for c in df.columns if not c.startswith("_")]].copy()
            top_n = min(5, len(display_df))

            fig_radar = go.Figure()
            for i, row in display_df.head(top_n).iterrows():
                values = [row[m] for m in metrics_radar]
                values.append(values[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_radar + [metrics_radar[0]],
                    name=row["Model"],
                    line=dict(color=DISCRETE_COLORS[i % len(DISCRETE_COLORS)]),
                ))
            fig_radar.update_layout(
                template=PLOTLY_TEMPLATE,
                polar=dict(radialaxis=dict(range=[0.0, 1.0])),
                title="Comparación Multimétrica (Top 5 Modelos)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("Recomendaciones")
        st.success(f"Se recomienda usar **{best['Model']}** para este problema.")

        if pt == "classification":
            if primary_value >= 0.95:
                st.info("Rendimiento excelente. El modelo es altamente discriminativo.")
            elif primary_value >= 0.85:
                st.info("Buen rendimiento. Considera optimizar hiperparámetros para mejorar.")
            else:
                st.warning("Rendimiento moderado. Considera más datos o feature engineering.")

        elif pt == "regression":
            if primary_value >= 0.85:
                st.info("El modelo explica más del 85% de la varianza. Excelente ajuste.")
            elif primary_value >= 0.70:
                st.info("Buen ajuste. Prueba con más features o transformaciones.")
            else:
                st.warning("R² bajo. El modelo puede estar subajustando (underfitting).")

        elif pt == "timeseries":
            mape = best.get("MAPE (%)")
            if pd.notna(mape) and mape < 5:
                st.info("MAPE < 5%: Forecasts muy precisos.")
            elif pd.notna(mape) and mape < 10:
                st.info("MAPE < 10%: Forecasts aceptables.")
            else:
                st.warning("MAPE alto. Considera más datos o ajustar períodos estacionales.")

        st.subheader("Próximos Pasos Sugeridos")
        st.markdown("""
        1. **Optimización de hiperparámetros**: Usa Grid Search o Random Search con Optuna.
        2. **Interpretabilidad**: Analiza SHAP values y feature importance.
        3. **Validación adicional**: Prueba con datos externos para confirmar generalización.
        4. **Monitoreo**: Implementa detección de model drift en producción.
        5. **AutoML**: Considera bibliotecas como AutoSklearn o H2O.ai para automatizar.
        """)
