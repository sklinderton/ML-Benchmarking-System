"""
streamlit_app.py - ML Benchmarking System
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
Melany Ramírez · Jason Barrantes · Junior Ramírez
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mlbenchmark.benchmarking import run_benchmark
from mlbenchmark.balancing import check_imbalance
from mlbenchmark.metrics import roc_curve_data
from mlbenchmark.threshold import optimize_threshold, threshold_analysis
from mlbenchmark.eda import analisisEDA

# ══════════════════════════════════════════════════════════════
# CONSTANTES VISUALES
# ══════════════════════════════════════════════════════════════
TMPL  = "plotly_dark"
C_MAIN = "Viridis"
C_REV  = "Plasma"
DISC   = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
          "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

# ══════════════════════════════════════════════════════════════
# UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════
def safe_df(df):
    """Sanitiza tipos mixtos; pasa Styler directamente."""
    try:
        from pandas.io.formats.style import Styler
        if isinstance(df, Styler):
            return df
    except ImportError:
        pass
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            try:
                out[col] = pd.to_numeric(out[col])
            except (ValueError, TypeError):
                out[col] = out[col].astype(str)
    return out


def show_df(df, **kwargs):
    kwargs.setdefault("width", "stretch")
    st.dataframe(safe_df(df), **kwargs)


def fmt(v):
    return f"{v:.4f}" if (v is not None and pd.notna(v)) else "N/A"


def style_table(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    float_cols = df.select_dtypes(include=["float"]).columns.tolist()
    s = df.style
    if float_cols:
        s = s.format({c: "{:.4f}" for c in float_cols})
    if num_cols:
        s = s.background_gradient(subset=num_cols, cmap="viridis")
        s = s.set_properties(subset=num_cols, **{"color": "white"})
    return s


# ══════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE MODELOS CON HIPERPARÁMETROS
# ══════════════════════════════════════════════════════════════
def build_models_with_hyperparams(problem_type, selected_models, hp, rs=42):
    """Construye modelos sklearn con los hiperparámetros dados por el usuario."""
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   RandomForestRegressor, GradientBoostingRegressor)
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    models = {}

    if problem_type == "classification":
        builders = {
            "Logistic Regression": lambda h: LogisticRegression(
                C=h.get("C", 1.0), max_iter=h.get("max_iter", 1000),
                solver=h.get("solver", "lbfgs"), random_state=rs),
            "Random Forest": lambda h: RandomForestClassifier(
                n_estimators=h.get("n_estimators", 100),
                max_depth=h.get("max_depth") or None,
                min_samples_split=h.get("min_samples_split", 2),
                max_features=h.get("max_features", "sqrt"), random_state=rs),
            "Decision Tree": lambda h: DecisionTreeClassifier(
                max_depth=h.get("max_depth") or None,
                min_samples_split=h.get("min_samples_split", 2),
                criterion=h.get("criterion", "gini"), random_state=rs),
            "SVM": lambda h: SVC(
                C=h.get("C", 1.0), kernel=h.get("kernel", "rbf"),
                gamma=h.get("gamma", "scale"), probability=True, random_state=rs),
            "K-Nearest Neighbors": lambda h: KNeighborsClassifier(
                n_neighbors=h.get("n_neighbors", 5),
                weights=h.get("weights", "uniform"),
                metric=h.get("metric", "minkowski")),
            "Naive Bayes": lambda h: GaussianNB(
                var_smoothing=h.get("var_smoothing", 1e-9)),
            "Gradient Boosting": lambda h: GradientBoostingClassifier(
                n_estimators=h.get("n_estimators", 100),
                learning_rate=h.get("learning_rate", 0.1),
                max_depth=h.get("max_depth", 3), random_state=rs),
        }
        try:
            from xgboost import XGBClassifier
            builders["XGBoost"] = lambda h: XGBClassifier(
                n_estimators=h.get("n_estimators", 100),
                learning_rate=h.get("learning_rate", 0.1),
                max_depth=h.get("max_depth", 6),
                subsample=h.get("subsample", 1.0),
                random_state=rs, eval_metric="logloss", verbosity=0)
        except ImportError:
            pass

    elif problem_type == "regression":
        builders = {
            "Ridge Regression": lambda h: Ridge(alpha=h.get("alpha", 1.0)),
            "Lasso Regression": lambda h: Lasso(
                alpha=h.get("alpha", 1.0), max_iter=2000),
            "Random Forest": lambda h: RandomForestRegressor(
                n_estimators=h.get("n_estimators", 100),
                max_depth=h.get("max_depth") or None,
                min_samples_split=h.get("min_samples_split", 2), random_state=rs),
            "Decision Tree": lambda h: DecisionTreeRegressor(
                max_depth=h.get("max_depth") or None,
                min_samples_split=h.get("min_samples_split", 2), random_state=rs),
            "SVR": lambda h: SVR(
                C=h.get("C", 1.0), kernel=h.get("kernel", "rbf"),
                gamma=h.get("gamma", "scale")),
            "K-Nearest Neighbors": lambda h: KNeighborsRegressor(
                n_neighbors=h.get("n_neighbors", 5),
                weights=h.get("weights", "uniform")),
            "Gradient Boosting": lambda h: GradientBoostingRegressor(
                n_estimators=h.get("n_estimators", 100),
                learning_rate=h.get("learning_rate", 0.1),
                max_depth=h.get("max_depth", 3), random_state=rs),
        }
        try:
            from xgboost import XGBRegressor
            builders["XGBoost"] = lambda h: XGBRegressor(
                n_estimators=h.get("n_estimators", 100),
                learning_rate=h.get("learning_rate", 0.1),
                max_depth=h.get("max_depth", 6), random_state=rs, verbosity=0)
        except ImportError:
            pass
    else:
        return {}

    for name in selected_models:
        if name in builders:
            models[name] = builders[name](hp.get(name, {}))
    return models


def build_ts_models_with_hyperparams(selected_models, hp, seasonal_periods):
    """Construye modelos de series de tiempo con hiperparámetros del usuario."""
    from mlbenchmark.models_timeseries import (
        HoltWintersModel, HoltWintersCalibrated,
        ARIMAModel, ARIMACalibrated, LSTMModel)
    ws_default = min(12, seasonal_periods)
    builders = {
        "Holt-Winters": lambda h: HoltWintersModel(
            seasonal_periods=seasonal_periods,
            trend=h.get("trend", "add"),
            seasonal=h.get("seasonal", "add")),
        "Holt-Winters Calibrado": lambda h: HoltWintersCalibrated(
            seasonal_periods=seasonal_periods),
        "ARIMA(1,1,1)": lambda h: ARIMAModel(
            order=(h.get("p", 1), h.get("d", 1), h.get("q", 1))),
        "ARIMA Calibrado": lambda h: ARIMACalibrated(
            max_p=h.get("max_p", 2),
            max_d=h.get("max_d", 2),
            max_q=h.get("max_q", 2)),
        "LSTM": lambda h: LSTMModel(
            units=h.get("units", 50),
            layers=h.get("layers", 2),
            epochs=h.get("epochs", 30),
            window_size=h.get("window_size", ws_default),
            dropout=h.get("dropout", 0.2),
            scale=False),
    }
    models = {}
    for name in selected_models:
        if name in builders:
            models[name] = builders[name](hp.get(name, {}))
    return models


# ══════════════════════════════════════════════════════════════
# CARGA DE DATASETS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_predefined_dataset(name, problem_type):
    """Carga datasets predefinidos. Retorna DataFrame completo + nombre de target."""
    from sklearn.datasets import fetch_california_housing, load_breast_cancer

    if problem_type == "Clasificación":
        if name == "Breast Cancer Wisconsin":
            d = load_breast_cancer()
            df = pd.DataFrame(d.data, columns=d.feature_names)
            df["target"] = d.target
            return df, "target"

        if name == "Credit Card Fraud (Simulado)":
            rng = np.random.RandomState(42)
            n, nf = 10000, 200
            X = np.vstack([rng.randn(n-nf, 20), rng.randn(nf, 20)+2.5])
            y = np.array([0]*(n-nf)+[1]*nf)
            idx = rng.permutation(n)
            cols = [f"feature_{i}" for i in range(20)]
            df = pd.DataFrame(X[idx], columns=cols)
            df["fraud"] = y[idx]
            return df, "fraud"

    elif problem_type == "Regresión":
        if name == "California Housing":
            d = fetch_california_housing()
            df = pd.DataFrame(d.data, columns=d.feature_names)
            df["price"] = d.target
            return df, "price"

    elif problem_type == "Series de Tiempo":
        if name == "Airline Passengers":
            p = [112,118,132,129,121,135,148,148,136,119,104,118,
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
                 417,391,419,461,472,535,622,606,508,461,390,432]
            return pd.Series(p, name="passengers"), None

    return None, None


def parse_uploaded_file(f, sep=",", decimal=".", use_idx=False):
    name = f.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f, sep=sep, decimal=decimal,
                             index_col=0 if use_idx else False)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f, index_col=0 if use_idx else False)
        elif name.endswith(".json"):
            df = pd.read_json(f)
        else:
            return None, "Formato no soportado. Usa CSV, Excel o JSON."
        df.columns = [str(c).strip() for c in df.columns]
        return df, None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Benchmarking System - BCD-7213 LEAD",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap');
html,body,[class*="css"],[class*="st-"],.stApp,.stApp*{font-family:'Poppins',sans-serif!important}
.stApp{background:#121212;color:#e0e0e0}
section[data-testid="stSidebar"]{background:#1e1e1e;border-right:1px solid #333}
section[data-testid="stSidebar"] *{color:#e0e0e0}
section[data-testid="stSidebar"] hr{border-color:#333}
h1,h2,h3,h4,h5,h6{color:#e0e0e0;font-weight:700}
p,li,span,label,div{color:#d6d6d6}
.main-header{background:linear-gradient(135deg,#121212 0%,#202020 50%,#1a1a1a 100%);
  padding:2rem;border-radius:12px;text-align:center;margin-bottom:2rem;
  border:1px solid #333;box-shadow:0 8px 18px rgba(0,0,0,.35)}
.main-header h1{color:#e0e0e0;font-size:2.2rem;margin:0}
.main-header p{color:#bdbdbd;margin:.5rem 0 0 0}
.best-model-banner{background:linear-gradient(135deg,#1e1e1e 0%,#333 60%,#2a2a2a 100%);
  padding:1.5rem;border-radius:12px;color:#e0e0e0;text-align:center;margin-bottom:1rem;
  border:1px solid #444;box-shadow:0 10px 22px rgba(0,0,0,.35)}
.best-model-banner h2,.best-model-banner h3,.best-model-banner p{color:#e0e0e0;margin:.25rem 0}
.stButton button{background:#333!important;color:#e0e0e0!important;border:1px solid #555!important;
  border-radius:12px!important;font-weight:600!important;padding:.55rem .9rem!important}
.stButton button:hover{background:#3a3a3a!important;border-color:#777!important}
div[data-testid="stDataFrame"]{background:#1e1e1e;border:1px solid #333;border-radius:12px;
  padding:.25rem;overflow:hidden}
hr{border-color:#333!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🤖 ML Benchmarking System</h1>
    <p>BCD-7213 Minería de Datos Avanzada · Universidad LEAD · I Cuatrimestre 2026</p>
    <p style="color:#e94560;font-size:.85rem;">Melany Ramírez · Jason Barrantes · Junior Ramírez</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuración")
    st.divider()

    problem_type = st.selectbox(
        "🎯 Tipo de Problema",
        ["Clasificación", "Regresión", "Series de Tiempo"],
        key="sb_problem_type",
    )

    st.subheader("📂 Fuente de Datos")
    dataset_options = {
        "Clasificación":    ["Breast Cancer Wisconsin", "Credit Card Fraud (Simulado)", "📤 Subir archivo"],
        "Regresión":        ["California Housing", "📤 Subir archivo"],
        "Series de Tiempo": ["Airline Passengers", "📤 Subir archivo"],
    }
    selected_dataset = st.selectbox("Dataset", dataset_options[problem_type], key="sb_dataset")

    uploaded_file = None
    csv_opts = {"sep": ",", "decimal": ".", "idx": False}
    if selected_dataset == "📤 Subir archivo":
        uploaded_file = st.file_uploader("Archivo CSV / Excel / JSON",
                                          type=["csv","xlsx","xls","json"],
                                          key="sb_uploader")
        if uploaded_file:
            with st.expander("⚙️ Opciones de parseo"):
                sep_raw = st.selectbox("Separador", [",",";","\\t","|"])
                decimal  = st.selectbox("Decimal", [".",","])
                use_idx  = st.checkbox("Primera col. como índice", False)
            sep_val = "\t" if sep_raw == "\\t" else sep_raw
            csv_opts = {"sep": sep_val, "decimal": decimal, "idx": use_idx}

    # MEJORA: campo URL para carga directa desde internet (CSV, JSON, scraping)
    st.subheader("🌐 O carga desde URL")
    dataset_url = st.text_input(
        "URL del dataset (CSV, JSON, Excel, o página web):",
        value="",
        placeholder="https://ejemplo.com/datos.csv",
        key="sb_dataset_url",
        help="Pega una URL directa a un CSV/JSON/Excel público, o una URL de página web para scraping automático.",
    )
    if dataset_url.strip():
        st.caption("✅ URL detectada — se cargará automáticamente al presionar **Cargar Dataset**")

    st.divider()

    threshold = 0.5; balancing = "none"; train_ratio = 0.8; seasonal_periods = 12
    test_size = 0.3; cv_folds = 5; scale_features_flag = True

    if problem_type != "Series de Tiempo":
        st.subheader("🔧 Parámetros")
        test_size           = st.slider("Test Set (%)", 10, 50, 30, 5, key="sb_test") / 100
        cv_folds            = st.slider("K-Folds", 3, 10, 5, key="sb_folds")
        scale_features_flag = st.checkbox("Escalar Features", True, key="sb_scale")
        if problem_type == "Clasificación":
            st.divider()
            threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05, key="sb_thr")
            balancing = st.selectbox("Balanceo",
                ["none","smote","undersample","combined"],
                format_func=lambda x: {"none":"Sin balanceo","smote":"SMOTE",
                    "undersample":"Under-sampling","combined":"Híbrido"}[x],
                key="sb_bal")
    else:
        st.subheader("📈 Series de Tiempo")
        train_ratio      = st.slider("Train Ratio (%)", 60, 90, 80, 5, key="sb_tr") / 100
        seasonal_periods = st.selectbox("Períodos Estacionales", [4,12,24,52], index=1, key="sb_sp")

    st.divider()
    st.caption("Configura y presiona **Cargar Dataset** para comenzar.")


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
_defaults = {
    "data_loaded": False, "benchmark_run": False,
    "results": None, "working_df": None, "target_col": None,
    "series": None, "hyperparams": {}, "selected_models": [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_explore, tab_config, tab_bench, tab_detail, tab_best, tab_wm, tab_nn, tab_ar = st.tabs([
    "🔍 Exploración & EDA",
    "⚙️ Configuración de Modelos",
    "🏆 Benchmarking",
    "📈 Resultados Detallados",
    "🥇 Mejor Modelo",
    "🌐 Web Mining",
    "🧠 Redes Neuronales",
    "🔗 Reglas de Asociación",
])


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 1 · EXPLORACIÓN & EDA                              ║
# ╚══════════════════════════════════════════════════════════╝
with tab_explore:
    st.header("🔍 Exploración, Limpieza y Análisis de Datos")

    # ── Botón Cargar ─────────────────────────────────────────
    if st.button("📥 Cargar Dataset", type="primary", use_container_width=True, key="btn_load"):
        with st.spinner("Cargando..."):
            err = None

            # MEJORA: rama URL — tiene prioridad sobre selector de dataset
            _url = st.session_state.get("sb_dataset_url", "").strip()
            if _url:
                try:
                    from mlbenchmark.web_mining import WebMiner
                    _miner = WebMiner(delay=0.5)
                    if problem_type == "Series de Tiempo":
                        # MEJORA: para TS intentar cargar como CSV numérico
                        _df_url = _miner.load_from_url(_url)
                        if _df_url.empty:
                            err = f"No se pudo cargar datos desde: {_url}"
                        else:
                            _num_cols = _df_url.select_dtypes(include="number").columns
                            if not len(_num_cols):
                                err = "No se encontraron columnas numéricas en la URL."
                            else:
                                st.session_state.series = _df_url[_num_cols[0]].dropna().values.astype(float)
                                st.session_state.working_df = None
                    else:
                        # MEJORA: para clasificación/regresión: scraping o CSV
                        _df_url, _log = _miner.scrape_with_log(url_base=_url, max_pages=2)
                        if _df_url.empty:
                            err = f"No se pudo cargar datos desde: {_url}"
                        else:
                            st.session_state.working_df = _df_url
                            # MEJORA: usar sugerir_target para detectar target automáticamente
                            from mlbenchmark.eda import analisisEDA
                            _eda_tmp = analisisEDA(_df_url)
                            _sugerencia = _eda_tmp.sugerir_target()
                            st.session_state.target_col = _sugerencia["columna_sugerida"]
                            st.session_state["_url_target_sugerido"] = _sugerencia
                            st.session_state.series = None
                except Exception as _e:
                    err = f"Error cargando URL: {_e}"

            elif selected_dataset == "📤 Subir archivo":
                if uploaded_file is None:
                    err = "No has subido ningún archivo."
                else:
                    df_raw, parse_err = parse_uploaded_file(
                        uploaded_file, csv_opts["sep"], csv_opts["decimal"], csv_opts["idx"])
                    if parse_err:
                        err = parse_err
                    else:
                        if problem_type == "Series de Tiempo":
                            num_cols = df_raw.select_dtypes(include="number").columns
                            if not len(num_cols):
                                err = "No hay columnas numéricas en el archivo."
                            else:
                                st.session_state.series = df_raw[num_cols[0]].dropna().values.astype(float)
                                st.session_state.working_df = None
                        else:
                            st.session_state.working_df = df_raw
                            # Target = última columna por defecto
                            st.session_state.target_col = df_raw.columns[-1]
                            st.session_state.series = None
            else:
                result = load_predefined_dataset(selected_dataset, problem_type)
                df_or_series, target = result
                if problem_type == "Series de Tiempo":
                    st.session_state.series = df_or_series.values.astype(float)
                    st.session_state.working_df = None
                else:
                    st.session_state.working_df = df_or_series
                    st.session_state.target_col = target
                    st.session_state.series = None

            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.data_loaded  = True
                st.session_state.benchmark_run = False
                st.session_state.results = None
                st.success("✅ Dataset cargado correctamente.")

    if not st.session_state.data_loaded:
        st.info("👈 Selecciona la fuente de datos y presiona **Cargar Dataset**.")
        if selected_dataset == "📤 Subir archivo":
            st.markdown("""
            **Instrucciones para archivos propios:**
            - **Clasificación / Regresión:** columnas de features + **target en la última columna**
              (o selecciónala tras cargar).
            - **Series de Tiempo:** una sola columna numérica con los valores de la serie.
            - Formatos: `.csv` · `.xlsx` · `.xls` · `.json`
            """)
        st.stop()

    # ──────────────────────────────────────────────────────────
    # SERIE DE TIEMPO
    # ──────────────────────────────────────────────────────────
    if problem_type == "Series de Tiempo":
        series = st.session_state.series
        st.metric("📅 Observaciones", len(series))

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Media",  f"{np.mean(series):.2f}")
        col_s2.metric("Mín",    f"{np.min(series):.2f}")
        col_s3.metric("Máx",    f"{np.max(series):.2f}")

        fig_ts = px.line(y=series, title="Serie Temporal",
                         labels={"index":"Tiempo","y":"Valor"}, template=TMPL)
        fig_ts.update_traces(line_color=DISC[0])
        st.plotly_chart(fig_ts, width="stretch")

        # Estadísticas básicas
        with st.expander("📋 Estadísticas Descriptivas"):
            ts_stats = pd.DataFrame({
                "Métrica": ["N","Media","Mediana","Desv. Std","Mínimo","Máximo",
                             "Q1 (25%)","Q3 (75%)","IQR"],
                "Valor": [len(series), np.mean(series), np.median(series),
                           np.std(series), np.min(series), np.max(series),
                           np.percentile(series,25), np.percentile(series,75),
                           np.percentile(series,75)-np.percentile(series,25)]
            })
            ts_stats["Valor"] = ts_stats["Valor"].round(4)
            show_df(ts_stats)

        # Descomposición básica
        with st.expander("📉 Histograma + Densidad"):
            import math
            fig_h, ax = plt.subplots(figsize=(10, 4), dpi=100)
            ax.hist(series, bins=30, color=DISC[0], edgecolor="black", alpha=0.7, density=True)
            try:
                import seaborn as sns
                sns.kdeplot(series, ax=ax, color=DISC[1], linewidth=2)
            except Exception:
                pass
            ax.set_title("Distribución de la Serie Temporal")
            ax.set_xlabel("Valor")
            ax.set_ylabel("Densidad")
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close("all")
        st.stop()

    # ──────────────────────────────────────────────────────────
    # CLASIFICACIÓN / REGRESIÓN
    # ──────────────────────────────────────────────────────────
    wdf = st.session_state.working_df

    # ── Selección de target (para archivos subidos) ───────────
    if selected_dataset == "📤 Subir archivo":
        with st.expander("🎯 Configuración de Columnas", expanded=True):
            cur_target = st.session_state.target_col or wdf.columns[-1]
            new_target = st.selectbox(
                "Columna Target (variable a predecir):",
                wdf.columns.tolist(),
                index=list(wdf.columns).index(cur_target) if cur_target in wdf.columns else len(wdf.columns)-1,
                key="exp_target_sel",
            )
            st.session_state.target_col = new_target
            st.info(f"Target: **{new_target}** · Features: {len(wdf.columns)-1} columnas")
            # MEJORA: mostrar sugerencia de target automática si viene de URL
            if "_url_target_sugerido" in st.session_state:
                _sug = st.session_state["_url_target_sugerido"]
                st.success(
                    f"🤖 **Target sugerido automáticamente:** `{_sug['columna_sugerida']}` "
                    f"— {_sug['razon']} (score={_sug['score']}) "
                    f"→ Problema: **{_sug['tipo_problema']}**"
                )

    target_col = st.session_state.target_col

    # ── KPI cards ─────────────────────────────────────────────
    n_dup  = wdf.duplicated().sum()
    n_null = int(wdf.isnull().sum().sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🗃️ Filas",       f"{len(wdf):,}")
    c2.metric("📐 Columnas",     len(wdf.columns))
    c3.metric("🔁 Duplicados",   n_dup)
    c4.metric("❓ Valores Nulos", n_null)

    # ── Vista previa ──────────────────────────────────────────
    # MEJORA: mostrar sugerencia de target automática con analisisEDA.sugerir_target()
    with st.expander("🤖 Detección Automática de Target (EDA)", expanded=False):
        try:
            from mlbenchmark.eda import analisisEDA as _EDA
            _eda_sug = _EDA(wdf)
            _sug_res = _eda_sug.sugerir_target()
            st.success(
                f"**Columna target sugerida:** `{_sug_res['columna_sugerida']}` "
                f"· Tipo de problema: **{_sug_res['tipo_problema']}** | "
                f"*Razón:* {_sug_res['razon']}"
            )
            show_df(_sug_res["detalle"].head(10))
        except Exception as _e:
            st.info(f"Análisis de target no disponible: {_e}")

    with st.expander("👁️ Vista Previa del Dataset", expanded=True):
        n_rows = st.slider("Filas a mostrar:", 5, 50, 10, key="exp_preview_rows")
        show_df(wdf.head(n_rows))

    # ══════════════════════════════════════════════════════
    # SECCIÓN: LIMPIEZA DE DATOS
    # ══════════════════════════════════════════════════════
    with st.expander("🧹 Limpieza de Datos", expanded=True):
        st.markdown("**Acciones de limpieza aplicadas al dataset en memoria.**")

        col_cl1, col_cl2 = st.columns(2)

        with col_cl1:
            if st.button("🗑️ Eliminar Filas Duplicadas", key="btn_dup"):
                antes = len(st.session_state.working_df)
                st.session_state.working_df = st.session_state.working_df.drop_duplicates()
                eliminadas = antes - len(st.session_state.working_df)
                if eliminadas:
                    st.success(f"Se eliminaron **{eliminadas}** filas duplicadas.")
                else:
                    st.info("No se encontraron duplicados.")
                st.rerun()

        with col_cl2:
            null_action = st.selectbox("Tratar valores nulos:",
                ["-- Selecciona acción --",
                 "Eliminar filas con nulos",
                 "Rellenar con Media (columnas numéricas)",
                 "Rellenar con Mediana (columnas numéricas)",
                 "Rellenar con Moda (todas las columnas)",
                 "Rellenar con cero"],
                key="null_action_sel")

            if st.button("✅ Aplicar tratamiento de nulos", key="btn_null"):
                wdf_tmp = st.session_state.working_df
                if null_action == "Eliminar filas con nulos":
                    st.session_state.working_df = wdf_tmp.dropna()
                    st.success("Filas con nulos eliminadas.")
                elif null_action == "Rellenar con Media (columnas numéricas)":
                    num_c = wdf_tmp.select_dtypes(include="number").columns
                    st.session_state.working_df = wdf_tmp.copy()
                    st.session_state.working_df[num_c] = wdf_tmp[num_c].fillna(wdf_tmp[num_c].mean())
                    st.success("Nulos numéricos rellenos con la media.")
                elif null_action == "Rellenar con Mediana (columnas numéricas)":
                    num_c = wdf_tmp.select_dtypes(include="number").columns
                    st.session_state.working_df = wdf_tmp.copy()
                    st.session_state.working_df[num_c] = wdf_tmp[num_c].fillna(wdf_tmp[num_c].median())
                    st.success("Nulos numéricos rellenos con la mediana.")
                elif null_action == "Rellenar con Moda (todas las columnas)":
                    tmp = wdf_tmp.copy()
                    for c in tmp.columns:
                        tmp[c] = tmp[c].fillna(tmp[c].mode().iloc[0] if not tmp[c].mode().empty else 0)
                    st.session_state.working_df = tmp
                    st.success("Nulos rellenos con la moda.")
                elif null_action == "Rellenar con cero":
                    st.session_state.working_df = wdf_tmp.fillna(0)
                    st.success("Nulos rellenos con 0.")
                else:
                    st.warning("Selecciona una acción primero.")
                st.rerun()

        st.markdown("---")
        cols_to_drop = st.multiselect(
            "🗂️ Seleccionar columnas a eliminar:",
            [c for c in wdf.columns if c != target_col],
            key="exp_drop_cols"
        )
        if st.button("❌ Eliminar columnas seleccionadas", key="btn_drop") and cols_to_drop:
            st.session_state.working_df = st.session_state.working_df.drop(columns=cols_to_drop, errors="ignore")
            st.success(f"Columnas eliminadas: {cols_to_drop}")
            st.rerun()

        if st.button("🔄 Revertir todos los cambios", key="btn_revert"):
            result = load_predefined_dataset(selected_dataset, problem_type)
            if selected_dataset != "📤 Subir archivo":
                df_or_s, tgt = result
                st.session_state.working_df = df_or_s
                st.session_state.target_col = tgt
            st.success("Dataset restaurado al estado original.")
            st.rerun()

    # Refrescar wdf después de posibles cambios
    wdf        = st.session_state.working_df
    target_col = st.session_state.target_col

    # ══════════════════════════════════════════════════════
    # SECCIÓN: ESTADÍSTICAS Y PERFILADO
    # ══════════════════════════════════════════════════════
    with st.expander("📋 Tipos de Datos y Valores Nulos"):
        tipo_df = pd.DataFrame({
            "Columna":   wdf.dtypes.index,
            "Tipo":      wdf.dtypes.values.astype(str),
            "No Nulos":  wdf.count().values,
            "Nulos":     wdf.isnull().sum().values,
            "% Nulos":   (wdf.isnull().sum() / len(wdf) * 100).round(2).values,
        })
        show_df(tipo_df)

    with st.expander("📐 Estadísticas Descriptivas"):
        num_df = wdf.select_dtypes(include="number")
        stat_sel = st.selectbox("Estadística:",
            ["Describe completo","Media","Mediana","Desv. Estándar","Mínimo","Máximo","Cuantiles"],
            key="stat_sel")
        if stat_sel == "Describe completo":
            show_df(num_df.describe().round(4))
        elif stat_sel == "Media":
            show_df(num_df.mean().round(4).to_frame("Media"))
        elif stat_sel == "Mediana":
            show_df(num_df.median().round(4).to_frame("Mediana"))
        elif stat_sel == "Desv. Estándar":
            show_df(num_df.std().round(4).to_frame("Desv. Std"))
        elif stat_sel == "Mínimo":
            show_df(num_df.min().round(4).to_frame("Mínimo"))
        elif stat_sel == "Máximo":
            show_df(num_df.max().round(4).to_frame("Máximo"))
        elif stat_sel == "Cuantiles":
            show_df(num_df.quantile([0, 0.25, 0.5, 0.75, 1]).round(4))

    with st.expander("🔢 Frecuencia de Valores por Columna"):
        col_freq = st.selectbox("Columna:", wdf.columns.tolist(), key="freq_col")
        vc = wdf[col_freq].value_counts().reset_index()
        vc.columns = ["Valor","Conteo"]
        vc["% del Total"] = (vc["Conteo"] / len(wdf) * 100).round(2)
        show_df(vc.head(30))
        if len(vc) <= 30:
            fig_vc = px.bar(vc, x="Valor", y="Conteo", color="Conteo",
                            color_continuous_scale=C_MAIN,
                            title=f"Frecuencia: {col_freq}", template=TMPL)
            st.plotly_chart(fig_vc, width="stretch")

    # ══════════════════════════════════════════════════════
    # SECCIÓN: DISTRIBUCIÓN DE CLASES (Clasificación)
    # ══════════════════════════════════════════════════════
    if problem_type == "Clasificación" and target_col in wdf.columns:
        with st.expander("🎯 Distribución del Target / Clases", expanded=True):
            y_vals = wdf[target_col]
            vc_t = y_vals.value_counts().reset_index()
            vc_t.columns = ["Clase","Conteo"]

            imb = check_imbalance(y_vals.values)
            m1,m2,m3 = st.columns(3)
            m1.metric("Clases únicas", len(vc_t))
            m2.metric("Ratio min/max", f"{imb['ratio']:.3f}")
            m3.metric("Severidad", imb["severity"])

            cc1, cc2 = st.columns(2)
            with cc1:
                fig_cb = px.bar(vc_t, x="Clase", y="Conteo", color="Conteo",
                                color_continuous_scale=C_MAIN,
                                title="Conteo por Clase", template=TMPL)
                st.plotly_chart(fig_cb, width="stretch")
            with cc2:
                fig_cp = px.pie(vc_t, values="Conteo", names="Clase",
                                color_discrete_sequence=DISC,
                                title="Proporción de Clases", template=TMPL)
                st.plotly_chart(fig_cp, width="stretch")

            if imb["is_imbalanced"]:
                st.warning(f"⚠️ Dataset desbalanceado (ratio={imb['ratio']:.3f}, "
                           f"severidad **{imb['severity']}**). Considera SMOTE en la configuración.")

    # ══════════════════════════════════════════════════════
    # SECCIÓN: VISUALIZACIONES EDA (matplotlib/seaborn)
    # ══════════════════════════════════════════════════════
    st.subheader("📉 Visualizaciones EDA")
    st.caption("Generadas con matplotlib/seaborn vía la clase analisisEDA (CRISP-DM)")

    viz_type = st.selectbox("Tipo de gráfico:", [
        "Correlación (Heatmap)",
        "Distribución + KDE",
        "Boxplots (detección de outliers)",
        "Densidad KDE",
        "Histogramas",
        "Dispersión por Pares (Pairplot)",
        "Distribución de la Clase (target)",
    ], key="viz_type_sel")

    if st.button("🎨 Generar Visualización", key="btn_viz"):
        with st.spinner("Generando gráfico..."):
            try:
                eda_obj = analisisEDA(wdf.select_dtypes(include=["number","object","category"]))
                fig_v = None

                if viz_type == "Correlación (Heatmap)":
                    fig_v = eda_obj.graficoCorrelacion()
                elif viz_type == "Distribución + KDE":
                    fig_v = eda_obj.distribucionVariables()
                elif viz_type == "Boxplots (detección de outliers)":
                    fig_v = eda_obj.graficoBoxplot()
                elif viz_type == "Densidad KDE":
                    fig_v = eda_obj.datosDensidad()
                elif viz_type == "Histogramas":
                    fig_v = eda_obj.histogramas()
                elif viz_type == "Dispersión por Pares (Pairplot)":
                    st.info("⏳ El pairplot puede tardar unos segundos...")
                    fig_v = eda_obj.graficosDispersion()
                elif viz_type == "Distribución de la Clase (target)":
                    if target_col and target_col in wdf.columns:
                        eda_full = analisisEDA(wdf)
                        fig_v = eda_full.histogramaClase(target_col)
                    else:
                        st.warning("Selecciona primero una columna target.")

                if fig_v:
                    st.pyplot(fig_v)
                    plt.close("all")
                else:
                    st.warning("No se pudo generar el gráfico (sin columnas numéricas o target no configurado).")
            except Exception as e:
                st.error(f"Error al generar el gráfico: {e}")

    # ══════════════════════════════════════════════════════
    # CORRELACIÓN INTERACTIVA (Plotly)
    # ══════════════════════════════════════════════════════
    with st.expander("🔗 Correlación Interactiva (Plotly)"):
        num_cols_corr = wdf.select_dtypes(include="number").columns
        if len(num_cols_corr) >= 2:
            corr_matrix = wdf[num_cols_corr].corr().round(3)
            fig_corr = px.imshow(corr_matrix, color_continuous_scale=C_MAIN,
                                  title="Matriz de Correlación", aspect="auto",
                                  text_auto=".2f", template=TMPL)
            st.plotly_chart(fig_corr, width="stretch")
        else:
            st.info("Se necesitan al menos 2 columnas numéricas para la correlación.")

    # Preparar X, y para benchmarking (se actualizan con working_df)
    if target_col and target_col in wdf.columns:
        st.session_state["_ready_X"] = wdf.drop(columns=[target_col])
        st.session_state["_ready_y"] = wdf[target_col]
    else:
        st.session_state["_ready_X"] = wdf
        st.session_state["_ready_y"] = None


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 2 · CONFIGURACIÓN DE MODELOS E HIPERPARÁMETROS     ║
# ╚══════════════════════════════════════════════════════════╝
with tab_config:
    st.header("⚙️ Configuración de Modelos e Hiperparámetros")
    st.info("Selecciona los modelos y configura sus hiperparámetros. "
            "Los parámetros del experimento (test size, K-Folds, balanceo) están en el sidebar.")

    if problem_type in ["Clasificación", "Regresión"]:
        all_clf = ["Logistic Regression","Random Forest","Decision Tree",
                   "SVM","K-Nearest Neighbors","Naive Bayes","Gradient Boosting","XGBoost"]
        all_reg = ["Ridge Regression","Lasso Regression","Random Forest","Decision Tree",
                   "SVR","K-Nearest Neighbors","Gradient Boosting","XGBoost"]
        pool = all_clf if problem_type == "Clasificación" else all_reg
        # Quitar XGBoost si no está instalado
        try:
            import xgboost  # noqa
        except ImportError:
            pool = [m for m in pool if m != "XGBoost"]

        st.subheader("🤖 Selección de Modelos")
        selected_models = st.multiselect("Modelos a comparar:", pool, default=pool, key="cfg_models")
        st.session_state.selected_models = selected_models

        st.subheader("🔧 Hiperparámetros por Modelo")
        st.caption("Expande cada modelo para ajustar sus parámetros. "
                   "Si no cambias nada, se usarán los valores predeterminados.")

        hp = st.session_state.get("hyperparams", {})

        for model_name in selected_models:
            with st.expander(f"⚙️ {model_name}", expanded=False):
                use_defaults = st.checkbox("Usar parámetros predeterminados",
                                            value=hp.get(model_name, {}).get("_defaults", True),
                                            key=f"def_{model_name}")
                if use_defaults:
                    hp[model_name] = {"_defaults": True}
                else:
                    mhp = {}
                    mhp["_defaults"] = False

                    # ─ Clasificación ─────────────────────────────────
                    if model_name == "Logistic Regression":
                        c1, c2 = st.columns(2)
                        mhp["C"] = c1.number_input("C (Regularización)", 0.001, 1000.0, 1.0, step=0.1,
                                                     key=f"lr_C_{model_name}")
                        mhp["max_iter"] = c2.number_input("Max iteraciones", 100, 10000, 1000, step=100,
                                                            key=f"lr_mi_{model_name}")
                        mhp["solver"] = st.selectbox("Solver",
                            ["lbfgs","liblinear","sag","saga"], key=f"lr_s_{model_name}")

                    elif model_name == "Random Forest":
                        c1, c2 = st.columns(2)
                        mhp["n_estimators"] = c1.slider("n_estimators", 10, 500, 100,
                                                          key=f"rf_ne_{model_name}")
                        mhp["max_depth"] = c2.slider("max_depth (0=None)", 0, 50, 0,
                                                       key=f"rf_md_{model_name}")
                        c3, c4 = st.columns(2)
                        mhp["min_samples_split"] = c3.slider("min_samples_split", 2, 20, 2,
                                                               key=f"rf_mss_{model_name}")
                        mhp["max_features"] = c4.selectbox("max_features",
                            ["sqrt","log2","None"], key=f"rf_mf_{model_name}")
                        if mhp["max_features"] == "None":
                            mhp["max_features"] = None

                    elif model_name == "Decision Tree":
                        c1, c2 = st.columns(2)
                        mhp["max_depth"] = c1.slider("max_depth (0=None)", 0, 50, 0,
                                                       key=f"dt_md_{model_name}")
                        mhp["min_samples_split"] = c2.slider("min_samples_split", 2, 20, 2,
                                                               key=f"dt_mss_{model_name}")
                        mhp["criterion"] = st.selectbox("criterion",
                            ["gini","entropy","log_loss"] if problem_type == "Clasificación"
                            else ["squared_error","friedman_mse","absolute_error","poisson"],
                            key=f"dt_cr_{model_name}")

                    elif model_name in ("SVM", "SVR"):
                        c1, c2 = st.columns(2)
                        mhp["C"] = c1.number_input("C", 0.001, 1000.0, 1.0, step=0.1,
                                                     key=f"svm_C_{model_name}")
                        mhp["kernel"] = c2.selectbox("Kernel",
                            ["rbf","linear","poly","sigmoid"], key=f"svm_k_{model_name}")
                        mhp["gamma"] = st.selectbox("Gamma",
                            ["scale","auto"], key=f"svm_g_{model_name}")

                    elif model_name == "K-Nearest Neighbors":
                        c1, c2 = st.columns(2)
                        mhp["n_neighbors"] = c1.slider("n_neighbors", 1, 50, 5,
                                                         key=f"knn_n_{model_name}")
                        mhp["weights"] = c2.selectbox("Weights",
                            ["uniform","distance"], key=f"knn_w_{model_name}")
                        mhp["metric"] = st.selectbox("Metric",
                            ["minkowski","euclidean","manhattan","chebyshev"],
                            key=f"knn_m_{model_name}")

                    elif model_name == "Naive Bayes":
                        mhp["var_smoothing"] = st.number_input(
                            "var_smoothing", 1e-12, 1e-3, 1e-9, format="%.2e",
                            key=f"nb_vs_{model_name}")

                    elif model_name in ("Gradient Boosting", "Ridge Regression",
                                        "Lasso Regression"):
                        c1, c2 = st.columns(2)
                        if model_name == "Gradient Boosting":
                            mhp["n_estimators"] = c1.slider("n_estimators", 50, 500, 100,
                                                              key=f"gb_ne_{model_name}")
                            mhp["learning_rate"] = c2.slider("learning_rate", 0.01, 1.0, 0.1,
                                                               key=f"gb_lr_{model_name}")
                            mhp["max_depth"] = st.slider("max_depth", 1, 15, 3,
                                                           key=f"gb_md_{model_name}")
                        else:  # Ridge / Lasso
                            mhp["alpha"] = c1.number_input("alpha (Regularización)",
                                0.0001, 1000.0, 1.0, step=0.1, key=f"rl_a_{model_name}")

                    elif model_name == "XGBoost":
                        c1, c2 = st.columns(2)
                        mhp["n_estimators"] = c1.slider("n_estimators", 50, 500, 100,
                                                          key=f"xgb_ne_{model_name}")
                        mhp["learning_rate"] = c2.slider("learning_rate", 0.01, 1.0, 0.1,
                                                           key=f"xgb_lr_{model_name}")
                        c3, c4 = st.columns(2)
                        mhp["max_depth"] = c3.slider("max_depth", 1, 15, 6,
                                                       key=f"xgb_md_{model_name}")
                        mhp["subsample"] = c4.slider("subsample", 0.4, 1.0, 1.0, step=0.05,
                                                       key=f"xgb_ss_{model_name}")

                    hp[model_name] = mhp

        st.session_state.hyperparams = hp

    else:
        # Series de tiempo
        ts_pool = ["Holt-Winters","Holt-Winters Calibrado",
                   "ARIMA(1,1,1)","ARIMA Calibrado","LSTM"]
        sel_ts = st.multiselect("Modelos:", ts_pool, default=ts_pool[:4], key="cfg_ts")
        st.session_state.selected_models = sel_ts

        st.subheader("🔧 Hiperparámetros")
        hp = st.session_state.get("hyperparams", {})

        for model_name in sel_ts:
            with st.expander(f"⚙️ {model_name}", expanded=False):
                use_def = st.checkbox("Usar predeterminados",
                                       value=hp.get(model_name, {}).get("_defaults", True),
                                       key=f"ts_def_{model_name}")
                if use_def:
                    hp[model_name] = {"_defaults": True}
                else:
                    mhp = {"_defaults": False}
                    if model_name == "Holt-Winters":
                        c1, c2 = st.columns(2)
                        mhp["trend"]    = c1.selectbox("trend",   ["add","mul","None"],
                                                        key=f"hw_tr_{model_name}")
                        mhp["seasonal"] = c2.selectbox("seasonal",["add","mul","None"],
                                                        key=f"hw_se_{model_name}")
                        if mhp["trend"]    == "None": mhp["trend"]    = None
                        if mhp["seasonal"] == "None": mhp["seasonal"] = None
                    elif model_name == "ARIMA(1,1,1)":
                        c1, c2, c3 = st.columns(3)
                        mhp["p"] = c1.slider("p (AR)", 0, 5, 1, key=f"ar_p_{model_name}")
                        mhp["d"] = c2.slider("d (I)",  0, 2, 1, key=f"ar_d_{model_name}")
                        mhp["q"] = c3.slider("q (MA)", 0, 5, 1, key=f"ar_q_{model_name}")
                    elif model_name == "ARIMA Calibrado":
                        c1, c2, c3 = st.columns(3)
                        mhp["max_p"] = c1.slider("max_p", 1, 5, 2, key=f"arc_mp_{model_name}")
                        mhp["max_d"] = c2.slider("max_d", 1, 2, 2, key=f"arc_md_{model_name}")
                        mhp["max_q"] = c3.slider("max_q", 1, 5, 2, key=f"arc_mq_{model_name}")
                    elif model_name == "LSTM":
                        c1, c2 = st.columns(2)
                        mhp["units"]       = c1.slider("Unidades LSTM", 16, 256, 50, step=16,
                                                         key=f"lstm_u_{model_name}")
                        mhp["layers"]      = c2.slider("Capas LSTM", 1, 4, 2,
                                                         key=f"lstm_l_{model_name}")
                        c3, c4 = st.columns(2)
                        mhp["epochs"]      = c3.slider("Épocas", 10, 200, 30, step=10,
                                                         key=f"lstm_e_{model_name}")
                        mhp["window_size"] = c4.slider("Ventana temporal", 5, 50, 12,
                                                         key=f"lstm_ws_{model_name}")
                        mhp["dropout"]     = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05,
                                                         key=f"lstm_do_{model_name}")
                    hp[model_name] = mhp
        st.session_state.hyperparams = hp

    st.divider()
    st.subheader("📋 Resumen de Configuración")
    cfg = {
        "Tipo de Problema": problem_type,
        "Dataset": selected_dataset,
    }
    if problem_type != "Series de Tiempo":
        cfg.update({"Test Size": f"{int(test_size*100)}%", "K-Folds": str(cv_folds),
                    "Escalar Features": str(scale_features_flag)})
        if problem_type == "Clasificación":
            cfg.update({"Threshold": str(threshold), "Balanceo": balancing})
    else:
        cfg.update({"Train Ratio": f"{int(train_ratio*100)}%",
                    "Períodos Estacionales": str(seasonal_periods)})
    show_df(pd.DataFrame({"Parámetro": cfg.keys(), "Valor": cfg.values()}))


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 3 · BENCHMARKING                                   ║
# ╚══════════════════════════════════════════════════════════╝
with tab_bench:
    st.header("🏆 Benchmarking de Modelos")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Primero carga el dataset en la pestaña **Exploración & EDA**.")
    else:
        if st.button("🚀 Iniciar Benchmarking", type="primary",
                     use_container_width=True, key="btn_bench"):
            with st.spinner("⏳ Entrenando y evaluando modelos..."):
                try:
                    hp   = st.session_state.get("hyperparams", {})
                    sels = st.session_state.get("selected_models", [])

                    if problem_type != "Series de Tiempo":
                        X_ready = st.session_state.get("_ready_X")
                        y_ready = st.session_state.get("_ready_y")
                        if X_ready is None or y_ready is None:
                            st.error("Confirma la configuración del dataset en Exploración.")
                            st.stop()

                        pt_key = "classification" if problem_type == "Clasificación" else "regression"
                        models_built = build_models_with_hyperparams(pt_key, sels, hp)

                        if not models_built:
                            st.error("Selecciona al menos un modelo en Configuración.")
                            st.stop()

                        result = run_benchmark(
                            problem_type=pt_key, X=X_ready, y=y_ready,
                            models=models_built,
                            test_size=test_size, cv_folds=cv_folds,
                            threshold=threshold if problem_type == "Clasificación" else 0.5,
                            balancing_technique=balancing if problem_type == "Clasificación" else "none",
                            scale=scale_features_flag,
                        )
                    else:
                        series = st.session_state.series
                        if series is None:
                            st.error("Carga la serie de tiempo primero.")
                            st.stop()
                        models_ts = build_ts_models_with_hyperparams(sels, hp, seasonal_periods)
                        if not models_ts:
                            st.error("Selecciona al menos un modelo en Configuración.")
                            st.stop()
                        result = run_benchmark(
                            problem_type="timeseries", series=series,
                            models=models_ts, seasonal_periods=seasonal_periods,
                            train_ratio=train_ratio,
                        )

                    st.session_state.results       = result
                    st.session_state.benchmark_run = True
                    st.success("✅ ¡Benchmarking completado!")

                except Exception as e:
                    import traceback
                    st.error(f"❌ Error: {e}")
                    st.code(traceback.format_exc())

        if st.session_state.benchmark_run and st.session_state.results:
            res   = st.session_state.results
            df    = res["results"]
            pt    = res["problem_type"]
            dcols = [c for c in df.columns if not c.startswith("_")]
            ddf   = df[dcols].copy()

            st.subheader("📊 Tabla Comparativa")
            show_df(style_table(ddf))

            st.subheader("📈 Comparación Visual")
            if pt == "classification":
                m = st.selectbox("Métrica:",
                    ["AUC-ROC","Accuracy","F1-Score","Recall","Precision","CV Mean"],
                    key="bench_metric")
                if m in ddf.columns:
                    fig_b = px.bar(ddf, x="Model", y=m, color=m,
                                   color_continuous_scale=C_MAIN,
                                   title=f"Comparación: {m}", text=m, template=TMPL)
                    fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig_b, width="stretch")

                if "CV Mean" in ddf.columns and "CV Std" in ddf.columns:
                    fig_cv = go.Figure(go.Bar(
                        x=ddf["Model"], y=ddf["CV Mean"],
                        error_y=dict(type="data", array=ddf["CV Std"]),
                        marker_color=DISC[0], name="CV Mean ± Std"))
                    fig_cv.update_layout(title="K-Fold CV (Mean ± Std)",
                                          xaxis_tickangle=-30, template=TMPL)
                    st.plotly_chart(fig_cv, width="stretch")

            elif pt == "regression":
                cc1, cc2 = st.columns(2)
                with cc1:
                    fig_r2 = px.bar(ddf, x="Model", y="R²", color="R²",
                                    color_continuous_scale=C_MAIN,
                                    title="R² Score", text="R²", template=TMPL)
                    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig_r2, width="stretch")
                with cc2:
                    fig_rm = px.bar(ddf, x="Model", y="RMSE", color="RMSE",
                                    color_continuous_scale=C_REV,
                                    title="RMSE", text="RMSE", template=TMPL)
                    fig_rm.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig_rm, width="stretch")

            elif pt == "timeseries":
                fig_tb = px.bar(ddf, x="Model", y="RMSE", color="RMSE",
                                color_continuous_scale=C_REV,
                                title="RMSE (menor=mejor)", text="RMSE", template=TMPL)
                fig_tb.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig_tb, width="stretch")

                train, test = res["train"], res["test"]
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(y=list(train), name="Train",
                                            line=dict(color=DISC[8])))
                fig_f.add_trace(go.Scatter(
                    x=list(range(len(train), len(train)+len(test))),
                    y=list(test), name="Real", line=dict(color=DISC[2], width=2)))
                for i, row in df.iterrows():
                    if row.get("_predictions") is not None:
                        fig_f.add_trace(go.Scatter(
                            x=list(range(len(train), len(train)+len(test))),
                            y=row["_predictions"], name=row["Model"],
                            line=dict(color=DISC[i % len(DISC)], dash="dash")))
                fig_f.update_layout(title="Forecasts vs Valores Reales", template=TMPL)
                st.plotly_chart(fig_f, width="stretch")


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 4 · RESULTADOS DETALLADOS                          ║
# ╚══════════════════════════════════════════════════════════╝
with tab_detail:
    st.header("📈 Resultados Detallados por Modelo")

    if not st.session_state.benchmark_run:
        st.warning("⚠️ Ejecuta el benchmarking primero.")
    else:
        res = st.session_state.results
        df  = res["results"]
        pt  = res["problem_type"]

        if pt == "classification":
            sel_m = st.selectbox("Modelo:", df["Model"].tolist(), key="det_clf_sel")
            row   = df[df["Model"] == sel_m].iloc[0]

            if row["Accuracy"] is None:
                st.error(f"❌ El modelo **{sel_m}** falló: `{row.get('_error','Error desconocido')}`")
                st.stop()

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Accuracy",  fmt(row["Accuracy"]))
            c2.metric("Precision", fmt(row["Precision"]))
            c3.metric("Recall",    fmt(row["Recall"]))
            c4.metric("F1-Score",  fmt(row["F1-Score"]))
            c5.metric("AUC-ROC",   fmt(row["AUC-ROC"]))

            y_test = res["y_test"]
            y_prob  = row["_y_prob"]
            col_r, col_c = st.columns(2)

            with col_r:
                if y_prob is not None:
                    try:
                        fpr, tpr, _ = roc_curve_data(y_test, y_prob)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy",
                            name=f"AUC={fmt(row['AUC-ROC'])}",
                            line=dict(color=DISC[0], width=2)))
                        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                            line=dict(dash="dash", color=DISC[9]), name="Aleatorio"))
                        fig_roc.update_layout(title="Curva ROC",
                            xaxis_title="FPR", yaxis_title="TPR", template=TMPL)
                        st.plotly_chart(fig_roc, width="stretch")
                    except Exception as e:
                        st.warning(str(e))

            with col_c:
                cm = row["_confusion_matrix"]
                if cm:
                    fig_cm = px.imshow(np.array(cm), text_auto=True,
                        x=["Neg","Pos"], y=["Neg","Pos"],
                        color_continuous_scale=C_MAIN, title="Matriz de Confusión",
                        labels=dict(x="Predicho", y="Real"))
                    fig_cm.update_layout(template=TMPL)
                    st.plotly_chart(fig_cm, width="stretch")

            st.subheader("🔄 Scores por Fold")
            cv_s = row["CV Scores"]
            if cv_s and len(cv_s) > 0:
                fdf = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv_s))],
                                     "AUC-ROC": cv_s})
                fig_cv = px.bar(fdf, x="Fold", y="AUC-ROC", color="AUC-ROC",
                                 color_continuous_scale=C_MAIN, template=TMPL,
                                 title=f"K-Fold | Mean={fmt(row['CV Mean'])} ± {fmt(row['CV Std'])}")
                fig_cv.add_hline(y=row["CV Mean"], line_dash="dash",
                                  line_color="white", annotation_text="Media")
                st.plotly_chart(fig_cv, width="stretch")

            st.subheader("⚖️ Análisis de Threshold")
            if y_prob is not None:
                thr_df = pd.DataFrame(threshold_analysis(y_test, y_prob))
                fig_t  = go.Figure()
                for cn, cl in zip(["accuracy","precision","recall","f1"], DISC[:4]):
                    fig_t.add_trace(go.Scatter(x=thr_df["threshold"], y=thr_df[cn],
                                               name=cn.capitalize(), mode="lines",
                                               line=dict(color=cl)))
                fig_t.update_layout(title="Métricas vs Threshold",
                                     xaxis_title="Threshold", yaxis_title="Score",
                                     template=TMPL)
                st.plotly_chart(fig_t, width="stretch")
                opt = optimize_threshold(y_test, y_prob, metric="f1")
                st.info(f"🎯 Threshold óptimo (F1): **{opt['optimal_threshold']}** "
                        f"— F1={opt['best_score']:.4f}")

        elif pt == "regression":
            sel = st.selectbox("Modelo:", df["Model"].tolist(), key="det_reg_sel")
            row = df[df["Model"] == sel].iloc[0]

            if row["R²"] is None:
                st.error(f"❌ El modelo **{sel}** falló: `{row.get('_error','')}`")
                st.stop()

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("R²",   fmt(row["R²"]))
            c2.metric("RMSE", fmt(row["RMSE"]))
            c3.metric("MAE",  fmt(row["MAE"]))
            c4.metric("CV Mean", f"{fmt(row['CV Mean (R²)'])} ± {fmt(row['CV Std'])}")

            cv_s = row["CV Scores"]
            if cv_s and len(cv_s) > 0:
                fdf = pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(len(cv_s))],"R²":cv_s})
                st.plotly_chart(px.bar(fdf, x="Fold", y="R²", color="R²",
                    color_continuous_scale=C_MAIN, template=TMPL,
                    title=f"K-Fold | Mean={fmt(row['CV Mean (R²)'])}"),
                    width="stretch")

        elif pt == "timeseries":
            sel = st.selectbox("Modelo:", df["Model"].tolist(), key="det_ts_sel")
            row = df[df["Model"] == sel].iloc[0]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("RMSE",    fmt(row["RMSE"]))
            c2.metric("MAE",     fmt(row["MAE"]))
            c3.metric("MSE",     fmt(row["MSE"]))
            c4.metric("MAPE (%)",f"{row['MAPE (%)']:.2f}" if row["MAPE (%)"] and pd.notna(row["MAPE (%)"]) else "N/A")

            if row.get("_predictions") is not None:
                train, test = res["train"], res["test"]
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(y=list(train), name="Train", line=dict(color=DISC[8])))
                fig_f.add_trace(go.Scatter(x=list(range(len(train),len(train)+len(test))),
                    y=list(test), name="Real", line=dict(color=DISC[2], width=2)))
                fig_f.add_trace(go.Scatter(x=list(range(len(train),len(train)+len(test))),
                    y=row["_predictions"], name="Predicción",
                    line=dict(color=DISC[0], dash="dash", width=2)))
                fig_f.update_layout(title=f"Forecast: {sel}", template=TMPL)
                st.plotly_chart(fig_f, width="stretch")


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 5 · MEJOR MODELO                                   ║
# ╚══════════════════════════════════════════════════════════╝
with tab_best:
    st.header("🥇 Mejor Modelo")

    if not st.session_state.benchmark_run:
        st.warning("⚠️ Ejecuta el benchmarking primero.")
    else:
        res  = st.session_state.results
        df   = res["results"]
        pt   = res["problem_type"]
        best = df.iloc[0]

        pm_map = {"classification":("AUC-ROC","AUC-ROC"),
                  "regression":    ("R²","R²"),
                  "timeseries":    ("RMSE","RMSE")}
        pm_col, pm_lbl = pm_map[pt]
        pm_val = best[pm_col]

        st.markdown(f"""
        <div class="best-model-banner">
            <h2>🏆 {best['Model']}</h2>
            <h3>{pm_lbl}: {fmt(pm_val)}</h3>
            <p>Mejor modelo según la métrica principal del benchmarking</p>
        </div>""", unsafe_allow_html=True)

        dcols = [c for c in df.columns if not c.startswith("_")]
        show_df(style_table(df[dcols].iloc[0:1]))

        if pt == "classification":
            st.subheader("🕸️ Radar Comparativo (Top 5)")
            met_r = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
            top5  = df[dcols].head(5)
            fig_r = go.Figure()
            for i, (_, row) in enumerate(top5.iterrows()):
                vals = [row[m] if row[m] is not None else 0 for m in met_r] + \
                       [row[met_r[0]] if row[met_r[0]] is not None else 0]
                fig_r.add_trace(go.Scatterpolar(r=vals, theta=met_r+[met_r[0]],
                    name=row["Model"], line=dict(color=DISC[i % len(DISC)])))
            fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                                 title="Top 5 — Comparación Multimétrica", template=TMPL)
            st.plotly_chart(fig_r, width="stretch")

        st.subheader("💡 Recomendaciones")
        st.success(f"✅ Se recomienda usar **{best['Model']}** para este problema.")

        if pt == "classification" and pm_val is not None:
            if pm_val >= 0.95:   st.info("🌟 AUC ≥ 0.95: Rendimiento excelente.")
            elif pm_val >= 0.85: st.info("👍 AUC ≥ 0.85: Buen rendimiento.")
            else:                st.warning("⚠️ AUC < 0.85: Considera más datos o feature engineering.")
        elif pt == "regression" and pm_val is not None:
            if pm_val >= 0.85:   st.info("🌟 R² ≥ 0.85: Excelente ajuste.")
            elif pm_val >= 0.70: st.info("👍 R² ≥ 0.70: Buen ajuste.")
            else:                st.warning("⚠️ R² < 0.70: El modelo puede estar subajustando.")
        elif pt == "timeseries":
            mape = best.get("MAPE (%)")
            if mape and pd.notna(mape) and mape < 5:    st.info("🌟 MAPE < 5%: Forecasts muy precisos.")
            elif mape and pd.notna(mape) and mape < 10: st.info("👍 MAPE < 10%: Forecasts aceptables.")
            else:                                        st.warning("⚠️ MAPE alto. Ajusta períodos estacionales.")

        st.subheader("🚀 Próximos Pasos")
        st.markdown("""
        1. **Optimización de hiperparámetros** — Grid Search / Random Search / Optuna
        2. **Interpretabilidad** — SHAP values y feature importance
        3. **Validación adicional** — Prueba con datos externos independientes
        4. **Monitoreo en producción** — Detecta model drift y degradación
        5. **AutoML** — Considera AutoSklearn o H2O.ai para automatizar la selección
        """)

# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 6 · WEB MINING                                     ║
# ╚══════════════════════════════════════════════════════════╝


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 6 · WEB MINING                                     ║
# ╚══════════════════════════════════════════════════════════╝
with tab_wm:
    st.header("🌐 Web Mining — Extracción y Análisis de Datos Web")
    st.markdown("""
    Extracción real de datos de e-commerce con **BeautifulSoup + requests**, 
    limpieza con **expresiones regulares** y análisis estadístico completo.
    El sistema soporta scraping real con paginación y dataset sintético offline.
    """)

    from mlbenchmark.web_mining import WebMiner

    # ── Configuración ─────────────────────────────────────────
    with st.expander("⚙️ Configuración del Scraping", expanded=True):
        col_wm0, col_wm1, col_wm2 = st.columns(3)
        wm_source = col_wm0.radio("Fuente de datos:",
            ["Dataset sintético (offline)", "Scraping real (requiere internet)"],
            key="wm_source_radio", label_visibility="collapsed", horizontal=False)
        wm_pages   = col_wm1.slider("Páginas a extraer:", 1, 5, 2, key="wm_pages_slider")
        wm_delay   = col_wm2.slider("Delay entre peticiones (s):", 0.5, 3.0, 1.0, 0.5,
                                     key="wm_delay_slider")

    run_wm = st.button("🚀 Ejecutar Web Mining", type="primary",
                        use_container_width=True, key="btn_wm_run")

    if run_wm:
        with st.spinner("⏳ Extrayendo y procesando datos..."):
            miner = WebMiner(delay=wm_delay)
            if "sintético" in wm_source:
                df_wm = miner.get_fallback_dataset()
                st.session_state["wm_df"]  = df_wm
                st.session_state["wm_src"] = "Dataset Sintético Outdoor (120 productos)"
                st.success(f"✅ Dataset sintético cargado — {len(df_wm)} productos.")
            else:
                with st.status("Scraping en progreso...", expanded=True) as status:
                    st.write("📡 Conectando al servidor...")
                    df_wm, log = miner.scrape_with_log(max_pages=wm_pages)
                    for msg in log:
                        st.write(msg)
                    if len(df_wm) == 0:
                        status.update(label="⚠️ Sin conexión — cargando dataset sintético",
                                      state="error")
                        df_wm = miner.get_fallback_dataset()
                        st.session_state["wm_src"] = "Dataset Sintético (fallback)"
                    else:
                        status.update(label=f"✅ Scraping completado", state="complete")
                        st.session_state["wm_src"] = f"Scraping real ({len(df_wm)} productos)"
                st.session_state["wm_df"] = df_wm

    if "wm_df" not in st.session_state:
        st.info("👆 Presiona **Ejecutar Web Mining** para comenzar.")
        st.markdown("""
        **El módulo de Web Mining:**
        - Descarga HTML real con `requests` + headers de navegador
        - Parsea estructura DOM con `BeautifulSoup`
        - Extrae: nombre, precio original, precio con descuento, estado
        - Limpia precios con expresiones regulares (`re.sub`)
        - Pagina automáticamente hasta N páginas
        - Fallback a dataset sintético cuando el sitio no responde
        """)
    else:
        df_wm  = st.session_state["wm_df"]
        wm_src = st.session_state.get("wm_src", "Dataset")
        miner  = WebMiner()
        stats  = miner.summary_stats(df_wm)

        # ── KPIs ─────────────────────────────────────────────
        st.caption(f"**Fuente:** {wm_src}")
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("📦 Productos",       f"{stats['total_productos']:,}")
        k2.metric("🏷️ Con Descuento",   f"{stats['con_descuento']} ({stats['tasa_descuento_%']}%)")
        k3.metric("🚫 Agotados",         stats['agotados'])
        k4.metric("💲 Precio Medio",    f"${stats['precio_promedio']:.2f}")
        k5.metric("📉 Precio Mín",      f"${stats['precio_minimo']:.2f}")
        k6.metric("📈 Precio Máx",      f"${stats['precio_maximo']:.2f}")

        # ── Datos crudos y limpios ────────────────────────────
        tab_raw, tab_clean, tab_viz, tab_regex = st.tabs(
            ["📄 Datos Extraídos", "🧹 Limpieza con Regex", "📊 Visualizaciones", "🔤 Demo Regex"])

        with tab_raw:
            cols_show = [c for c in df_wm.columns if not c.startswith("_")]
            n_show = st.slider("Filas a mostrar:", 10, min(200, len(df_wm)), 30,
                                key="wm_rows")
            show_df(df_wm[cols_show].head(n_show))
            st.caption(f"Total: {len(df_wm)} productos · {len(cols_show)} columnas")

        with tab_clean:
            st.markdown("### Proceso de Limpieza con Expresiones Regulares")
            st.markdown("""
            El precio original viene como texto crudo del HTML, por ejemplo:
            `"\\nPrecio original\\n\\n            $59.90\\n          "`.
            
            Se aplican estos patrones regex en orden:
            """)
            reglas_regex = [
                (r"(?i)precio\s+(original|actual)", "Elimina texto 'Precio original/actual'"),
                (r"[$,\\n\\r\\t]", "Elimina símbolos $, comas y espacios en blanco"),
                (r"^\\s+|\\s+$", "Elimina espacios al inicio y final (strip)"),
            ]
            for pat, desc in reglas_regex:
                st.code(f're.sub(r"{pat}", "", texto)  # {desc}', language="python")

            # Mostrar ejemplo concreto con datos reales
            if "PrecioOriginal" in df_wm.columns:
                st.markdown("**Estadísticas de precios tras limpieza:**")
                col_p = "PrecioFinal" if "PrecioFinal" in df_wm.columns else "PrecioOriginal"
                price_stats = df_wm[col_p].describe().round(2)
                show_df(price_stats.to_frame("Precio ($)"))

            if "TieneDescuento" in df_wm.columns and df_wm["TieneDescuento"].sum() > 0:
                st.markdown("**Productos con descuento (muestra):**")
                disc_sample = df_wm[df_wm["TieneDescuento"]==1][
                    [c for c in ["Nombre","Categoria","PrecioOriginal",
                                  "PrecioDescuento","PctDescuento"] if c in df_wm.columns]
                ].head(10)
                show_df(disc_sample)

        with tab_viz:
            col_p = "PrecioFinal" if "PrecioFinal" in df_wm.columns else "PrecioOriginal"

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig_h = px.histogram(df_wm, x=col_p, nbins=30,
                                      title="Distribución de Precios ($)",
                                      color_discrete_sequence=[DISC[0]], template=TMPL)
                fig_h.update_layout(xaxis_title="Precio ($)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_h, width="stretch")

            with col_v2:
                if "TieneDescuento" in df_wm.columns:
                    vc = df_wm["TieneDescuento"].map({1:"Con Descuento",0:"Precio Normal"})\
                           .value_counts().reset_index()
                    vc.columns = ["Estado","Conteo"]
                    fig_pie = px.pie(vc, values="Conteo", names="Estado",
                                     title="Distribución de Descuentos",
                                     color_discrete_sequence=DISC, template=TMPL)
                    st.plotly_chart(fig_pie, width="stretch")

            if "Categoria" in df_wm.columns:
                cat_g = df_wm.groupby("Categoria").agg(
                    Productos=(col_p,"count"),
                    Precio_Medio=(col_p,"mean"),
                    Precio_Max=(col_p,"max")
                ).round(2).reset_index().sort_values("Precio_Medio", ascending=False)

                fig_cat = px.bar(cat_g, x="Categoria", y="Precio_Medio",
                                  color="Precio_Medio", color_continuous_scale=C_MAIN,
                                  text="Productos", title="Precio Medio por Categoría",
                                  template=TMPL)
                fig_cat.update_traces(textposition="outside")
                st.plotly_chart(fig_cat, width="stretch")

                # Top 10 más caros
                top10 = df_wm.nlargest(10, col_p)[
                    [c for c in ["Nombre","Categoria","PrecioOriginal",col_p]
                     if c in df_wm.columns]]
                st.subheader("🏆 Top 10 Productos más Caros")
                show_df(top10)

            if "Marca" in df_wm.columns:
                marca_g = df_wm.groupby("Marca")[col_p].mean().round(2)\
                               .sort_values(ascending=False).reset_index()
                marca_g.columns = ["Marca","Precio Medio ($)"]
                fig_marca = px.bar(marca_g.head(10), x="Marca", y="Precio Medio ($)",
                                    color="Precio Medio ($)",
                                    color_continuous_scale=C_MAIN,
                                    title="Top 10 Marcas por Precio Medio",
                                    template=TMPL)
                st.plotly_chart(fig_marca, width="stretch")

            if "PctDescuento" in df_wm.columns and df_wm["PctDescuento"].sum() > 0:
                fig_disc = px.histogram(df_wm[df_wm["PctDescuento"]>0],
                                         x="PctDescuento", nbins=20,
                                         title="Distribución del % de Descuento",
                                         color_discrete_sequence=[DISC[2]], template=TMPL)
                fig_disc.update_layout(xaxis_title="Descuento (%)")
                st.plotly_chart(fig_disc, width="stretch")

        with tab_regex:
            st.markdown("### 🔤 Motor de Extracción con Expresiones Regulares")
            st.markdown("Prueba patrones regex sobre los nombres/textos extraídos.")

            col_re1, col_re2 = st.columns([2,1])
            sample_text = col_re1.text_area(
                "Texto de muestra:",
                value="\n".join(df_wm["Nombre"].head(8).tolist()) if "Nombre" in df_wm.columns
                      else "Ejemplo de texto scrapeado",
                height=200, key="wm_regex_text")
            pattern = col_re2.text_input("Patrón regex:", value=r"\b[A-Z][a-z]+\b",
                                           key="wm_regex_pat")
            flag_i  = col_re2.checkbox("Ignorar mayúsculas (IGNORECASE)", True, key="wm_flag_i")
            flag_m  = col_re2.checkbox("Multilínea (MULTILINE)", False, key="wm_flag_m")

            if st.button("🔍 Aplicar Regex", key="btn_regex"):
                import re
                flags = 0
                if flag_i: flags |= re.IGNORECASE
                if flag_m: flags |= re.MULTILINE
                try:
                    matches = re.findall(pattern, sample_text, flags)
                    st.success(f"✅ {len(matches)} coincidencias encontradas:")
                    st.code("\n".join(str(m) for m in matches[:50]))
                    if len(matches) > 50:
                        st.caption(f"… y {len(matches)-50} más.")
                except re.error as e:
                    st.error(f"Patrón inválido: {e}")

            st.markdown("**Patrones de ejemplo útiles:**")
            ejemplos = {
                r"\$[\d,]+\.?\d*":          "Precios (ej. $59.90)",
                r"\b[A-Z][a-z]+ [A-Z][a-z]+\b": "Nombres propios (dos palabras)",
                r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?": "Números con formato",
                r"https?://[^\s]+":          "URLs",
                r"\b\w{10,}\b":              "Palabras largas (≥ 10 letras)",
            }
            for pat, desc in ejemplos.items():
                st.code(f'r"{pat}"  →  {desc}', language="python")


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 7 · REDES NEURONALES                               ║
# ╚══════════════════════════════════════════════════════════╝
with tab_nn:
    st.header("🧠 Redes Neuronales — 5 Arquitecturas")
    st.markdown("""
    Benchmarking de **5 tipos de redes neuronales** entrenadas sobre los productos
    del Web Mining para predecir descuento (*clasificación*) o precio (*regresión*).
    """)

    from mlbenchmark.neural_networks import (
        benchmark_neural_networks, ARCHITECTURES_INFO, _check_keras)

    # ── Descripción de arquitecturas ──────────────────────────
    with st.expander("📐 Las 5 Arquitecturas Implementadas"):
        for name, info in ARCHITECTURES_INFO.items():
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"**{name}**")
            c2.markdown(
                f"🏗️ *{info['Tipo']}* · 🔧 {info['Framework']}  \n"
                f"`{info['Capas']}`  \n"
                f"*Regularización:* {info['Regularización']} · *Uso:* {info['Uso']}")
            st.divider()

    if not _check_keras():
        st.warning("⚠️ TensorFlow no detectado — se usarán 5 variantes de MLP sklearn como "
                   "fallback. Para activar Keras: `uv pip install tensorflow`")

    # ── Configuración ─────────────────────────────────────────
    st.subheader("⚙️ Configuración")
    col_n1, col_n2, col_n3, col_n4 = st.columns(4)

    # Usamos prefijo "_nn_" para evitar conflicto con session_state
    nn_task_sel  = col_n1.selectbox("Tarea:",
        ["Clasificación — TieneDescuento", "Regresión — PrecioFinal"],
        key="nn_task_sel")
    nn_epochs    = col_n2.slider("Épocas (Keras):", 10, 150, 40, 5, key="nn_epochs_sl")
    nn_testsize  = col_n3.slider("Test size (%):", 10, 40, 20, 5, key="nn_test_sl") / 100
    nn_dataset   = col_n4.selectbox("Dataset:",
        ["Productos Web Mining", "Breast Cancer (clasificación)", "California Housing (regresión)"],
        key="nn_dataset_sel")

    run_nn = st.button("🚀 Entrenar las 5 Redes Neuronales", type="primary",
                        use_container_width=True, key="btn_nn_run")

    if run_nn:
        # Determinar task
        _task = "classification" if "Clasificación" in nn_task_sel else "regression"

        with st.spinner("⏳ Entrenando... (1-5 min según hardware y épocas)"):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler, LabelEncoder

                # ── Fuente de datos ───────────────────────────
                if "Breast Cancer" in nn_dataset:
                    from sklearn.datasets import load_breast_cancer
                    X_raw, y_raw = load_breast_cancer(return_X_y=True)
                    _task = "classification"

                elif "California" in nn_dataset:
                    from sklearn.datasets import fetch_california_housing
                    data = fetch_california_housing()
                    X_raw, y_raw = data.data, data.target
                    _task = "regression"

                else:
                    # Web Mining dataset
                    if "wm_df" not in st.session_state:
                        st.error("⚠️ Primero ejecuta el **Web Mining** (Tab 6).")
                        st.stop()

                    df_wm_nn = st.session_state["wm_df"].copy()
                    _target  = ("TieneDescuento" if _task == "classification"
                                else "PrecioFinal")

                    if _target not in df_wm_nn.columns:
                        st.error(f"Columna '{_target}' no encontrada. "
                                 "Ejecuta Web Mining primero.")
                        st.stop()

                    # Encoding categóricas
                    for col in df_wm_nn.select_dtypes(include="object").columns:
                        if col != "Nombre":
                            df_wm_nn[col] = LabelEncoder().fit_transform(
                                df_wm_nn[col].astype(str))

                    num_cols = df_wm_nn.select_dtypes(include=[float, int]).columns.tolist()
                    feat_cols = [c for c in num_cols if c != _target]
                    X_raw = df_wm_nn[feat_cols].fillna(0).values
                    y_raw = df_wm_nn[_target].values

                # ── Preprocesamiento ──────────────────────────
                sc = StandardScaler()
                X_s = sc.fit_transform(X_raw)

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_s, y_raw, test_size=nn_testsize, random_state=42,
                    stratify=y_raw if _task == "classification" else None)

                # ── Benchmarking ──────────────────────────────
                df_nn_res = benchmark_neural_networks(
                    X_tr, X_te, y_tr, y_te,
                    task=_task,
                    epochs=nn_epochs,
                    random_state=42,
                )

                # Guardar con claves que NO conflicten con widgets
                st.session_state["_nn_results"] = df_nn_res
                st.session_state["_nn_task"]    = _task
                st.session_state["_nn_dataset"] = nn_dataset
                st.session_state["_nn_n_train"] = len(X_tr)
                st.session_state["_nn_n_test"]  = len(X_te)
                st.success("✅ ¡Benchmarking de redes neuronales completado!")

            except Exception as e:
                import traceback
                st.error(f"❌ {e}")
                st.code(traceback.format_exc())

    if "_nn_results" not in st.session_state:
        st.info("👆 Configura y presiona **Entrenar las 5 Redes Neuronales**.")
    else:
        df_nn_res = st.session_state["_nn_results"]
        _task     = st.session_state["_nn_task"]

        st.subheader("📊 Resultados Comparativos")
        st.caption(
            f"Dataset: **{st.session_state['_nn_dataset']}** · "
            f"Train: {st.session_state['_nn_n_train']} · "
            f"Test: {st.session_state['_nn_n_test']}")

        # Tabla limpia (sin columnas _error)
        display_nn = df_nn_res.drop(columns=["_error"], errors="ignore").copy()
        show_df(display_nn)

        # ── Gráfico comparativo ───────────────────────────────
        st.subheader("📈 Comparación Visual")

        if _task == "classification":
            avail_metrics = [m for m in ["Accuracy","F1-Score","AUC-ROC"]
                             if m in display_nn.columns]
            nn_metric = st.selectbox("Métrica:", avail_metrics, key="nn_metric_sel")
            df_plot = display_nn[display_nn[nn_metric].notna()].copy()

            if not df_plot.empty:
                fig_nn = px.bar(df_plot, x="Modelo", y=nn_metric,
                                color=nn_metric, color_continuous_scale=C_MAIN,
                                title=f"Redes Neuronales — {nn_metric}",
                                text=nn_metric, template=TMPL)
                fig_nn.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_nn.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig_nn, width="stretch")

            # Radar
            met_r = [m for m in ["Accuracy","F1-Score","AUC-ROC"]
                     if m in display_nn.columns]
            df_rad = display_nn.dropna(subset=met_r[:1])
            if len(df_rad) > 1 and len(met_r) >= 2:
                st.subheader("🕸️ Radar Comparativo")
                fig_rad = go.Figure()
                for i, (_, row) in enumerate(df_rad.iterrows()):
                    vals = [float(row.get(m, 0) or 0) for m in met_r]
                    vals_closed = vals + [vals[0]]
                    theta_closed = met_r + [met_r[0]]
                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals_closed, theta=theta_closed,
                        name=row["Modelo"],
                        line=dict(color=DISC[i % len(DISC)])))
                fig_rad.update_layout(
                    polar=dict(radialaxis=dict(range=[0,1])),
                    title="Comparación Multimétrica", template=TMPL)
                st.plotly_chart(fig_rad, width="stretch")

        else:  # regression
            met_reg = [m for m in ["R²","RMSE","MAE"] if m in display_nn.columns]
            df_plot = display_nn.dropna(subset=["R²"] if "R²" in display_nn.columns else met_reg[:1])

            if not df_plot.empty and "R²" in df_plot.columns:
                cc1, cc2 = st.columns(2)
                with cc1:
                    fig_r2 = px.bar(df_plot, x="Modelo", y="R²",
                                    color="R²", color_continuous_scale=C_MAIN,
                                    title="R² por Arquitectura", text="R²", template=TMPL)
                    fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                    st.plotly_chart(fig_r2, width="stretch")
                with cc2:
                    if "RMSE" in df_plot.columns:
                        fig_rm = px.bar(df_plot, x="Modelo", y="RMSE",
                                        color="RMSE", color_continuous_scale=C_REV,
                                        title="RMSE por Arquitectura", text="RMSE",
                                        template=TMPL)
                        fig_rm.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                        st.plotly_chart(fig_rm, width="stretch")

        # ── Sección Autoencoder ───────────────────────────────
        ae_row = display_nn[display_nn["Modelo"].str.contains("Autoencoder", na=False)]
        if not ae_row.empty:
            st.subheader("🔍 Autoencoder — Detección de Anomalías")
            row_ae = ae_row.iloc[0]
            ca1,ca2,ca3,ca4 = st.columns(4)
            ca1.metric("Error Reconstrucción",
                       f"{row_ae.get('Error Reconstrucción (medio)', 'N/A'):.6f}"
                       if row_ae.get('Error Reconstrucción (medio)') is not None else "N/A")
            ca2.metric("Threshold",
                       f"{row_ae.get('Threshold', 'N/A'):.6f}"
                       if row_ae.get('Threshold') is not None else "N/A")
            ca3.metric("Anomalías detectadas",
                       row_ae.get("Anomalías detectadas", "N/A"))
            ca4.metric("% Anomalías",
                       f"{row_ae.get('% Anomalías', 0):.2f}%")
            st.info("El Autoencoder aprende la distribución normal. Las muestras con alto "
                    "error de reconstrucción (> threshold) se etiquetan como anomalías "
                    "(p.ej. precios atípicos o características inusuales).")


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 8 · REGLAS DE ASOCIACIÓN                           ║
# ╚══════════════════════════════════════════════════════════╝
with tab_ar:
    st.header("🔗 Reglas de Asociación — Market Basket Analysis")
    st.markdown("""
    Minería de **patrones frecuentes de compra** con el algoritmo **Apriori** (mlxtend).
    Genera reglas *antecedente → consecuente* para recomendaciones de cross-selling.
    """)

    from mlbenchmark.association_rules import (
        AssociationRulesMiner, generate_synthetic_transactions, load_groceries_dataset)

    # ── Marco teórico rápido ──────────────────────────────────
    with st.expander("📚 Conceptos Clave"):
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Soporte",    "P(A ∩ B)",    "Frecuencia del itemset")
        cc2.metric("Confianza",  "P(B|A)",       "P de B dado A")
        cc3.metric("Lift",       "conf / P(B)", "> 1 → asociación real")
        st.markdown("""
        - **Soporte mínimo:** filtra itemsets poco frecuentes
        - **Confianza mínima:** filtra reglas poco fiables
        - **Lift > 1:** confirma que la regla no es aleatoria
        """)

    # ── Fuente y parámetros ───────────────────────────────────
    col_s1, col_s2 = st.columns([2,1])
    ar_source = col_s1.radio("Fuente de transacciones:",
        ["Productos outdoor (Web Mining)", "Dataset Groceries (público)"],
        key="ar_source_radio", label_visibility="collapsed", horizontal=True)
    ar_n_trans = col_s2.slider("Transacciones sintéticas:", 500, 5000, 2000, 100,
                                key="ar_n_trans_sl",
                                help="Solo para fuente de productos outdoor")

    col_p1, col_p2, col_p3 = st.columns(3)
    ar_min_sup  = col_p1.slider("Soporte mínimo:",  0.005, 0.20, 0.03, 0.005,
                                 format="%.3f", key="ar_sup_sl")
    ar_min_conf = col_p2.slider("Confianza mínima:", 0.05, 0.90, 0.20, 0.05, key="ar_conf_sl")
    ar_min_lift = col_p3.slider("Lift mínimo:",      1.0,  5.0,  1.0,  0.1,  key="ar_lift_sl")

    run_ar = st.button("🚀 Ejecutar Análisis de Reglas de Asociación",
                        type="primary", use_container_width=True, key="btn_ar_run")

    if run_ar:
        with st.spinner("⏳ Calculando itemsets y reglas..."):
            try:
                ar_miner = AssociationRulesMiner()

                if "Groceries" in ar_source:
                    df_groc = load_groceries_dataset()
                    ar_miner.fit_from_dataframe(df_groc, "id_compra", "item")
                    _ar_label = "Dataset Groceries (público)"
                else:
                    if "wm_df" not in st.session_state:
                        st.error("⚠️ Primero ejecuta el **Web Mining** (Tab 6).")
                        st.stop()
                    trans = generate_synthetic_transactions(
                        st.session_state["wm_df"], n_transacciones=ar_n_trans)
                    ar_miner.fit(trans)
                    _ar_label = f"Productos Outdoor ({ar_n_trans} transacciones sintéticas)"

                df_its   = ar_miner.get_frequent_itemsets(min_support=ar_min_sup)
                df_rules = ar_miner.get_rules(
                    min_confidence=ar_min_conf,
                    min_lift=ar_min_lift,
                    min_support=ar_min_sup)

                st.session_state["_ar_miner"]  = ar_miner
                st.session_state["_ar_its"]    = df_its
                st.session_state["_ar_rules"]  = df_rules
                st.session_state["_ar_label"]  = _ar_label

                if df_rules.empty:
                    st.warning(f"Se encontraron {len(df_its)} itemsets frecuentes, "
                               "pero no se generaron reglas con los parámetros actuales. "
                               "Prueba reducir la confianza o el lift mínimo.")
                else:
                    st.success(f"✅ {len(df_rules)} reglas generadas desde {len(df_its)} itemsets.")

            except Exception as e:
                import traceback
                st.error(f"❌ {e}")
                st.code(traceback.format_exc())

    if "_ar_miner" not in st.session_state:
        st.info("👆 Configura los parámetros y presiona **Ejecutar**.")
    else:
        ar_miner  = st.session_state["_ar_miner"]
        df_its    = st.session_state["_ar_its"]
        df_rules  = st.session_state["_ar_rules"]
        _ar_label = st.session_state["_ar_label"]
        _ar_sum   = ar_miner.summary()

        st.caption(f"**Fuente:** {_ar_label}")

        # KPIs
        ks1,ks2,ks3,ks4,ks5,ks6 = st.columns(6)
        ks1.metric("Transacciones",     f"{_ar_sum.get('total_transacciones',0):,}")
        ks2.metric("Ítems únicos",       _ar_sum.get("total_items_unicos",0))
        ks3.metric("Itemsets frecuentes",_ar_sum.get("itemsets_frecuentes",0))
        ks4.metric("Reglas",             _ar_sum.get("reglas_generadas",0))
        ks5.metric("Confianza media",    _ar_sum.get("confianza_media",0))
        ks6.metric("Lift medio",         _ar_sum.get("lift_medio",0))

        dist_ar = ar_miner.distribution_stats()
        if dist_ar:
            with st.expander("📉 Distribución de ítems por transacción"):
                d1,d2,d3,d4 = st.columns(4)
                d1.metric("Media",        dist_ar["media_items_por_transaccion"])
                d2.metric("Máximo",       dist_ar["max_items"])
                d3.metric("Mínimo",       dist_ar["min_items"])
                d4.metric("≥ 5 ítems",   dist_ar["transacciones_5_mas_items"])

        tab_its, tab_rules, tab_viz, tab_rec = st.tabs([
            "🧺 Itemsets", "📋 Reglas", "📊 Visualizaciones", "💡 Recomendaciones"])

        with tab_its:
            st.subheader("Itemsets Frecuentes")
            if df_its.empty:
                st.warning("No se encontraron itemsets. Reduce el soporte mínimo.")
            else:
                top_its = ar_miner.top_itemsets(n=20, min_size=1)
                show_df(top_its)
                top2 = ar_miner.top_itemsets(n=15, min_size=2)
                if not top2.empty:
                    fig_its = px.bar(top2, x="support", y="itemsets_str",
                                     orientation="h", color="support",
                                     color_continuous_scale=C_MAIN,
                                     title="Top Itemsets (≥ 2 ítems) por Soporte",
                                     template=TMPL)
                    fig_its.update_layout(yaxis_title="", xaxis_title="Soporte")
                    st.plotly_chart(fig_its, width="stretch")

        with tab_rules:
            if df_rules.empty:
                st.warning("No hay reglas. Reduce confianza o lift mínimo.")
            else:
                st.subheader(f"Top Reglas (ordenadas por Lift)")
                top_r = ar_miner.top_rules(n=25, by="lift")
                show_df(top_r)

                st.subheader("Top Reglas por Confianza")
                top_c = ar_miner.top_rules(n=25, by="confidence")
                show_df(top_c)

        with tab_viz:
            if df_rules.empty:
                st.warning("No hay reglas para visualizar.")
            else:
                # Scatter support × confidence × lift
                st.subheader("Mapa Support × Confidence × Lift")
                fig_sc = px.scatter(
                    df_rules,
                    x="support", y="confidence",
                    size="lift", color="lift",
                    color_continuous_scale=C_MAIN,
                    hover_data=["antecedents_str","consequents_str","lift"],
                    title="Reglas: Soporte vs Confianza (tamaño = Lift)",
                    template=TMPL)
                fig_sc.update_layout(xaxis_title="Soporte", yaxis_title="Confianza")
                st.plotly_chart(fig_sc, width="stretch")

                # Distribuciones
                col_dist1, col_dist2, col_dist3 = st.columns(3)
                with col_dist1:
                    fig_sup = px.histogram(df_rules, x="support", nbins=20,
                                            title="Distribución del Soporte",
                                            color_discrete_sequence=[DISC[0]], template=TMPL)
                    st.plotly_chart(fig_sup, width="stretch")
                with col_dist2:
                    fig_con = px.histogram(df_rules, x="confidence", nbins=20,
                                            title="Distribución de Confianza",
                                            color_discrete_sequence=[DISC[1]], template=TMPL)
                    st.plotly_chart(fig_con, width="stretch")
                with col_dist3:
                    fig_lft = px.histogram(df_rules, x="lift", nbins=20,
                                            title="Distribución del Lift",
                                            color_discrete_sequence=[DISC[2]], template=TMPL)
                    st.plotly_chart(fig_lft, width="stretch")

                # Heatmap
                if len(df_rules) <= 500:
                    st.subheader("Heatmap de Confianza")
                    top_ants = df_rules["antecedents_str"].value_counts().head(8).index.tolist()
                    hm_df = df_rules[df_rules["antecedents_str"].isin(top_ants)]
                    if not hm_df.empty:
                        pivot = hm_df.pivot_table(
                            index="antecedents_str",
                            columns="consequents_str",
                            values="confidence", aggfunc="max").fillna(0)
                        if not pivot.empty and pivot.shape[1] <= 30:
                            fig_hm = px.imshow(pivot,
                                               color_continuous_scale=C_MAIN,
                                               title="Confianza: Antecedente → Consecuente",
                                               aspect="auto", template=TMPL,
                                               text_auto=".2f")
                            st.plotly_chart(fig_hm, width="stretch")

        with tab_rec:
            st.subheader("💡 Motor de Recomendaciones")
            if df_rules.empty:
                st.warning("Genera reglas primero.")
            else:
                all_items = sorted(set(
                    item for ant in df_rules["antecedents"] for item in ant))

                col_r1, col_r2 = st.columns([3,1])
                sel_item = col_r1.selectbox("Si el cliente compra:", all_items,
                                             key="ar_item_sel")
                top_n_r  = col_r2.slider("Top N:", 1, 15, 5, key="ar_topn_sl")

                recs = ar_miner.recommend(sel_item, top_n=top_n_r)
                if not recs.empty:
                    st.success(f"**Cuando el cliente compra `{sel_item}`, "
                                f"también podría comprar:**")
                    cc_r1, cc_r2 = st.columns([1,2])
                    with cc_r1:
                        show_df(recs)
                    with cc_r2:
                        fig_rec = px.bar(recs, x="Confianza", y="Recomendación",
                                         orientation="h", color="Lift",
                                         color_continuous_scale=C_MAIN,
                                         title=f"Recomendaciones para: {sel_item}",
                                         text="Confianza", template=TMPL)
                        fig_rec.update_traces(texttemplate="%{text:.3f}")
                        st.plotly_chart(fig_rec, width="stretch")
                else:
                    st.info(f"No hay reglas para `{sel_item}`. "
                            "Reduce la confianza o el lift mínimo.")

                # Búsqueda libre
                st.divider()
                st.subheader("🔎 Buscar Reglas por Ítem")
                col_f1, col_f2 = st.columns([3,1])
                filter_item = col_f1.text_input("Ítem a buscar:", key="ar_filter_inp")
                filter_side = col_f2.selectbox("Posición:",
                    ["antecedents","consequents","any"], key="ar_filter_side")
                if filter_item.strip():
                    filtered = ar_miner.filter_rules_by_item(filter_item.strip(), filter_side)
                    if not filtered.empty:
                        show_df(filtered)
                    else:
                        st.info(f"No se encontraron reglas con `{filter_item}`.")