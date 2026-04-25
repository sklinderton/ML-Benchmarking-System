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

        if name == "IBM Telco Customer Churn":
            # Buscar CSV local en el mismo directorio del repo
            import os
            _csv_candidates = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
                os.path.join(os.path.dirname(__file__),
                             "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
            ]
            for _p in _csv_candidates:
                if os.path.exists(_p):
                    df = pd.read_csv(_p)
                    return df, "Churn"
            # Fallback: descargar desde URL pública
            _url = (
                "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
                "master/data/Telco-Customer-Churn.csv"
            )
            try:
                df = pd.read_csv(_url)
                return df, "Churn"
            except Exception:
                pass
            return None, None

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
        "Clasificación":    ["Breast Cancer Wisconsin", "Credit Card Fraud (Simulado)",
                             "IBM Telco Customer Churn", "📤 Subir archivo"],
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
tab_explore, tab_config, tab_bench, tab_detail, tab_best, tab_wm, tab_nn, tab_ar, tab_churn = st.tabs([
    "🔍 Exploración & EDA",
    "⚙️ Configuración de Modelos",
    "🏆 Benchmarking",
    "📈 Resultados Detallados",
    "🥇 Mejor Modelo",
    "🌐 Web Mining",
    "🧠 Redes Neuronales",
    "🔗 Reglas de Asociación",
    "📡 Análisis Churn (CRISP-DM)",
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

    _nn_dataset_opts = ["Productos Web Mining", "Breast Cancer (clasificación)", "California Housing (regresión)"]
    _user_df_loaded = st.session_state.get("data_loaded") and st.session_state.get("working_df") is not None
    if _user_df_loaded:
        _nn_dataset_opts.insert(0, "📂 Dataset cargado (Tab 1)")
    nn_dataset = col_n4.selectbox("Dataset:", _nn_dataset_opts, key="nn_dataset_sel")

    # Opciones de tarea: dependen del dataset seleccionado
    if "Dataset cargado" in nn_dataset and _user_df_loaded:
        _tgt_name = st.session_state.target_col or ""
        _nn_task_opts = [
            f"Clasificación — {_tgt_name}",
            f"Regresión — {_tgt_name}",
        ]
        _nn_task_help = (
            f"Target detectado: **{_tgt_name}**. "
            "Elige Clasificación si es una categoría/etiqueta discreta, "
            "o Regresión si es un valor numérico continuo."
        )
    elif "Breast Cancer" in nn_dataset:
        _nn_task_opts = ["Clasificación — target (maligno/benigno)"]
        _nn_task_help = "Breast Cancer es siempre clasificación binaria."
    elif "California" in nn_dataset:
        _nn_task_opts = ["Regresión — price (precio de vivienda)"]
        _nn_task_help = "California Housing es siempre regresión."
    else:
        _nn_task_opts = ["Clasificación — TieneDescuento", "Regresión — PrecioFinal"]
        _nn_task_help = "Columnas del dataset Web Mining."

    nn_task_sel = col_n1.selectbox("Tarea:", _nn_task_opts, key="nn_task_sel",
                                    help=_nn_task_help)
    nn_epochs   = col_n2.slider("Épocas (Keras):", 10, 150, 40, 5, key="nn_epochs_sl")
    nn_testsize = col_n3.slider("Test size (%):", 10, 40, 20, 5, key="nn_test_sl") / 100

    run_nn = st.button("🚀 Entrenar las 5 Redes Neuronales", type="primary",
                        use_container_width=True, key="btn_nn_run")

    if run_nn:
        # Determinar task a partir del texto de la opción seleccionada
        _task = "classification" if "Clasificación" in nn_task_sel else "regression"

        with st.spinner("⏳ Entrenando... (1-5 min según hardware y épocas)"):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler, LabelEncoder

                # ── Fuente de datos ───────────────────────────
                if "Dataset cargado" in nn_dataset:
                    df_loaded = st.session_state.working_df.copy()
                    _tgt = st.session_state.target_col
                    if _tgt not in df_loaded.columns:
                        st.error(f"Columna target '{_tgt}' no encontrada. Ve a Tab 1 y configura el target.")
                        st.stop()
                    from sklearn.preprocessing import LabelEncoder as _LE
                    for _c in df_loaded.select_dtypes(include="object").columns:
                        df_loaded[_c] = _LE().fit_transform(df_loaded[_c].astype(str))
                    _feat_cols = [c for c in df_loaded.columns if c != _tgt]
                    X_raw = df_loaded[_feat_cols].fillna(0).values
                    y_raw = df_loaded[_tgt].values
                    # Auto-detect task if not overridden by user
                    _n_unique = len(set(y_raw))
                    if _n_unique <= 20:
                        _task = "classification"
                    else:
                        _task = "regression"

                elif "Breast Cancer" in nn_dataset:
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
    _ar_sources = ["Productos outdoor (Web Mining)", "Dataset Groceries (público)"]
    if st.session_state.get("data_loaded") and st.session_state.get("working_df") is not None:
        _ar_sources.insert(0, "📂 Dataset cargado (Tab 1)")

    col_s1, col_s2 = st.columns([2,1])
    ar_source = col_s1.radio("Fuente de transacciones:",
        _ar_sources,
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

                if "Dataset cargado" in ar_source:
                    df_ar_loaded = st.session_state.working_df.copy()
                    _tgt_ar = st.session_state.target_col
                    _n_rows = len(df_ar_loaded)

                    # ── Seleccionar columnas útiles para asociación ──────
                    # Solo columnas con cardinalidad baja (≤ 20 valores únicos
                    # O ≤ 10% de las filas) para garantizar soporte suficiente.
                    _MAX_CARD = min(20, max(5, int(_n_rows * 0.10)))
                    _cat_cols = [
                        c for c in df_ar_loaded.select_dtypes(include="object").columns
                        if c != _tgt_ar and df_ar_loaded[c].nunique() <= _MAX_CARD
                    ]

                    # Para columnas numéricas: discretizar en 4 bins (Bajo/Medio/Alto/Muy Alto)
                    _num_cols_to_bin = [
                        c for c in df_ar_loaded.select_dtypes(include=[np.number]).columns
                        if c != _tgt_ar and df_ar_loaded[c].nunique() > 2
                    ]
                    _bin_labels = ["Bajo", "Medio", "Alto", "Muy Alto"]
                    for _bc in _num_cols_to_bin:
                        try:
                            df_ar_loaded[f"{_bc}_bin"] = pd.qcut(
                                df_ar_loaded[_bc], q=4,
                                labels=[f"{_bc}:{l}" for l in _bin_labels],
                                duplicates="drop"
                            ).astype(str)
                            _cat_cols.append(f"{_bc}_bin")
                        except Exception:
                            pass

                    # Columnas booleanas / binarias directas
                    _bool_cols = [
                        c for c in df_ar_loaded.select_dtypes(include=[np.number]).columns
                        if c != _tgt_ar and df_ar_loaded[c].nunique() == 2
                    ]
                    for _bc in _bool_cols:
                        df_ar_loaded[f"{_bc}_flag"] = df_ar_loaded[_bc].map(
                            lambda v: f"{_bc}:Sí" if v else f"{_bc}:No"
                        )
                        _cat_cols.append(f"{_bc}_flag")

                    _item_cols = list(dict.fromkeys(_cat_cols))  # deduplicar preservando orden

                    if not _item_cols:
                        st.error(
                            "No se encontraron columnas adecuadas para reglas de asociación. "
                            "El dataset necesita al menos una columna categórica o numérica discreta."
                        )
                        st.stop()

                    # ── Límite de ítems únicos para evitar OOM con Apriori ──
                    # Apriori es exponencial en nº de ítems únicos.
                    # Con >50 columnas la matriz binaria crece a cientos de miles
                    # de combinaciones y provoca MemoryError.
                    _MAX_ITEM_COLS = 20
                    if len(_item_cols) > _MAX_ITEM_COLS:
                        # Priorizar columnas con menor cardinalidad (más frecuentes → más soporte)
                        _item_cols = sorted(
                            _item_cols,
                            key=lambda c: df_ar_loaded[c].nunique() if c in df_ar_loaded.columns else 999
                        )[:_MAX_ITEM_COLS]
                        st.warning(
                            f"⚠️ El dataset tiene muchas columnas. "
                            f"Se usan las **{_MAX_ITEM_COLS} columnas con menor cardinalidad** "
                            f"para evitar errores de memoria. "
                            f"Apriori funciona mejor con ≤ 50 ítems únicos."
                        )

                    # ── Construir transacciones ──────────────────────────
                    # Cada fila = una transacción; cada celda no-nula = un ítem
                    def _row_to_items(row, cols):
                        items = []
                        for c in cols:
                            v = row[c]
                            if pd.notna(v) and str(v) not in ("nan", "None", ""):
                                items.append(str(v) if ":" in str(v) else f"{c}:{str(v)}")
                        return items

                    _trans_loaded = [
                        _row_to_items(row, _item_cols)
                        for _, row in df_ar_loaded[_item_cols].iterrows()
                    ]
                    _trans_loaded = [t for t in _trans_loaded if len(t) >= 2]

                    if not _trans_loaded:
                        st.error("No se pudieron extraer transacciones con al menos 2 ítems.")
                        st.stop()

                    # Verificar ítems únicos totales antes de llamar a Apriori
                    _unique_items = len({item for t in _trans_loaded for item in t})
                    if _unique_items > 100:
                        st.error(
                            f"❌ Demasiados ítems únicos ({_unique_items}). "
                            "Apriori requiere ≤ 100 ítems únicos para no agotar la memoria. "
                            "Sube el **Soporte mínimo** o usa un dataset con menos columnas."
                        )
                        st.stop()

                    ar_miner.fit(_trans_loaded)
                    _ar_label = (
                        f"Dataset cargado — {len(_trans_loaded)} transacciones · "
                        f"{_unique_items} ítems únicos · "
                        f"{len(_item_cols)} columnas"
                    )

                elif "Groceries" in ar_source:
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


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 9 · ANÁLISIS CHURN — CRISP-DM COMPLETO            ║
# ╚══════════════════════════════════════════════════════════╝
with tab_churn:
    st.header("📡 Análisis de Churn — Pipeline CRISP-DM Completo")
    st.markdown("""
    Pipeline integrado para el artículo científico **BCD-6210**.
    Cubre las 6 fases CRISP-DM aplicadas al dataset IBM Telco Customer Churn:
    Clasificación · K-Means Clustering · Reglas de Asociación · Redes Neuronales · Prueba Wilcoxon.
    """)

    # ── Tracker de fases CRISP-DM ─────────────────────────────
    st.markdown("""
    <style>
    .phase-badge{display:inline-block;padding:3px 10px;border-radius:12px;
      font-size:.75rem;font-weight:700;margin-right:6px;margin-bottom:4px}
    .ph1{background:#1a3a4a;color:#7ecfea}.ph2{background:#1a3a2a;color:#7eea9b}
    .ph3{background:#3a2a1a;color:#eab97e}.ph4{background:#2a1a3a;color:#bf9eea}
    .ph5{background:#3a1a1a;color:#ea7e7e}.ph6{background:#1a1a3a;color:#7e9eea}
    </style>
    <div>
      <span class="phase-badge ph1">Fase 1 · Business Understanding</span>
      <span class="phase-badge ph2">Fase 2 · Data Understanding</span>
      <span class="phase-badge ph3">Fase 3 · Data Preparation</span>
      <span class="phase-badge ph4">Fase 4 · Modeling</span>
      <span class="phase-badge ph5">Fase 5 · Evaluation</span>
      <span class="phase-badge ph6">Fase 6 · Deployment</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Pipeline state ────────────────────────────────────────
    _churn_state_keys = [
        "churn_loaded", "churn_clf_run", "churn_km_run",
        "churn_ar_run", "churn_nn_run", "churn_stat_run",
    ]
    for _k in _churn_state_keys:
        if _k not in st.session_state:
            st.session_state[_k] = False

    with st.sidebar:
        st.divider()
        st.subheader("📡 Pipeline Churn")
        st.checkbox("✅ Datos cargados",     value=st.session_state.churn_loaded,     disabled=True, key="_ch_s1")
        st.checkbox("✅ Clasificación",      value=st.session_state.churn_clf_run,    disabled=True, key="_ch_s2")
        st.checkbox("✅ K-Means",            value=st.session_state.churn_km_run,     disabled=True, key="_ch_s3")
        st.checkbox("✅ Reglas Asociación",  value=st.session_state.churn_ar_run,     disabled=True, key="_ch_s4")
        st.checkbox("✅ Redes Neuronales",   value=st.session_state.churn_nn_run,     disabled=True, key="_ch_s5")
        st.checkbox("✅ Prueba Estadística", value=st.session_state.churn_stat_run,   disabled=True, key="_ch_s6")

    # ══════════════════════════════════════════════════════════
    # FASE 1-2: CARGA Y COMPRENSIÓN DEL DATASET
    # ══════════════════════════════════════════════════════════
    st.markdown('<span class="phase-badge ph2">Fase 2 · Data Understanding</span>', unsafe_allow_html=True)
    st.subheader("📂 Carga del Dataset Telco")

    _churn_src = st.radio(
        "Fuente:",
        ["📂 Dataset cargado (Tab 1)", "📤 Subir CSV Telco", "⬇️ Usar IBM Telco predefinido"],
        horizontal=True, key="churn_src_radio",
    )

    _churn_upfile = None
    if _churn_src == "📤 Subir CSV Telco":
        _churn_upfile = st.file_uploader(
            "WA_Fn-UseC_-Telco-Customer-Churn.csv",
            type=["csv"], key="churn_up_file",
        )

    if st.button("📥 Cargar Dataset Telco", type="primary", use_container_width=True, key="btn_churn_load"):
        with st.spinner("Cargando..."):
            try:
                if _churn_src == "📂 Dataset cargado (Tab 1)":
                    if not st.session_state.get("data_loaded") or st.session_state.working_df is None:
                        st.error("Primero carga un dataset en Tab 1.")
                        st.stop()
                    _df_raw = st.session_state.working_df.copy()
                elif _churn_src == "📤 Subir CSV Telco":
                    if _churn_upfile is None:
                        st.error("Sube el archivo CSV primero.")
                        st.stop()
                    _df_raw = pd.read_csv(_churn_upfile)
                else:
                    _df_raw, _ = load_predefined_dataset("IBM Telco Customer Churn", "Clasificación")
                    if _df_raw is None:
                        st.error("No se pudo cargar el dataset predefinido. Sube el CSV manualmente.")
                        st.stop()

                st.session_state["churn_df_raw"] = _df_raw
                st.session_state.churn_loaded = True
                st.success(f"✅ Cargado: {_df_raw.shape[0]:,} filas × {_df_raw.shape[1]} columnas")
            except Exception as _e:
                st.error(f"❌ {_e}")

    if not st.session_state.churn_loaded or "churn_df_raw" not in st.session_state:
        st.info("👆 Carga el dataset para continuar.")
        st.stop()

    _df_raw = st.session_state["churn_df_raw"]

    with st.expander("🔍 Vista previa del dataset crudo"):
        show_df(_df_raw.head(10))
        c_r1, c_r2, c_r3 = st.columns(3)
        c_r1.metric("Filas", f"{_df_raw.shape[0]:,}")
        c_r2.metric("Columnas", _df_raw.shape[1])
        _churn_col_raw = next((c for c in _df_raw.columns if c.lower() == "churn"), None)
        if _churn_col_raw:
            _churn_pct = (_df_raw[_churn_col_raw].map(
                lambda v: 1 if str(v).strip().lower() in ("yes","1","true") else 0
            ).mean() * 100)
            c_r3.metric("Tasa de Churn", f"{_churn_pct:.1f}%")
        _null_sum = _df_raw.isnull().sum().sum() + (_df_raw == " ").sum().sum()
        st.caption(f"Valores faltantes / blancos detectados: **{_null_sum}**")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # FASE 3: PREPROCESAMIENTO
    # ══════════════════════════════════════════════════════════
    st.markdown('<span class="phase-badge ph3">Fase 3 · Data Preparation</span>', unsafe_allow_html=True)
    st.subheader("⚙️ Preprocesamiento")

    _churn_col_name = st.text_input("Nombre de la columna target (Churn):",
                                     value="Churn", key="churn_col_name_inp")
    _encode_method = st.radio("Codificación categórica:",
                               ["One-Hot Encoding (OHE)", "Label Encoding"],
                               horizontal=True, key="churn_encode_radio")
    _impute_method = st.selectbox("Estrategia de imputación:",
                                   ["median", "mean", "mode", "constant"],
                                   key="churn_impute_sel")
    _scale_method  = st.selectbox("Escalado:",
                                   ["standard (Z-score)", "minmax (0-1)", "none"],
                                   key="churn_scale_sel")

    if st.button("🔧 Aplicar Preprocesamiento", use_container_width=True, key="btn_churn_prep"):
        with st.spinner("Preprocesando..."):
            try:
                from mlbenchmark.preprocessing import (
                    impute_missing, encode_categorical, encode_categorical_ohe, scale_features)
                from mlbenchmark.clustering import preprocess_telco

                _df_proc = preprocess_telco(_df_raw) if _churn_col_name == "Churn" else _df_raw.copy()

                if _encode_method == "Label Encoding":
                    _df_proc, _ = encode_categorical(_df_proc)

                st.session_state["churn_df_proc"] = _df_proc
                st.success(f"✅ Preprocesado: {_df_proc.shape[0]:,} × {_df_proc.shape[1]} columnas")
                show_df(_df_proc.head(5))
            except Exception as _e:
                import traceback
                st.error(f"❌ {_e}")
                st.code(traceback.format_exc())

    if "churn_df_proc" not in st.session_state:
        st.info("👆 Aplica el preprocesamiento para continuar.")
        st.stop()

    _df_proc = st.session_state["churn_df_proc"]
    _target_churn = _churn_col_name if _churn_col_name in _df_proc.columns else _df_proc.columns[-1]

    st.divider()

    # ══════════════════════════════════════════════════════════
    # FASE 4a: CLASIFICACIÓN — Benchmarking con hiperparámetros
    # ══════════════════════════════════════════════════════════
    st.markdown('<span class="phase-badge ph4">Fase 4 · Modeling — Clasificación</span>', unsafe_allow_html=True)
    st.subheader("🤖 Benchmarking de Clasificadores")

    _clf_models_all = ["Logistic Regression", "Random Forest", "Decision Tree",
                       "SVM", "K-Nearest Neighbors", "Naive Bayes", "Gradient Boosting", "XGBoost"]
    try:
        import xgboost  # noqa
    except ImportError:
        _clf_models_all = [m for m in _clf_models_all if m != "XGBoost"]

    _sel_clf = st.multiselect("Modelos:", _clf_models_all, default=_clf_models_all, key="churn_clf_sel")

    _churn_hp = st.session_state.get("churn_hp", {})
    for _mn in _sel_clf:
        with st.expander(f"⚙️ {_mn}", expanded=False):
            _use_def = st.checkbox("Usar predeterminados", value=_churn_hp.get(_mn, {}).get("_defaults", True),
                                    key=f"churn_def_{_mn}")
            if _use_def:
                _churn_hp[_mn] = {"_defaults": True}
            else:
                _mhp = {"_defaults": False}
                if _mn == "Logistic Regression":
                    _c1, _c2 = st.columns(2)
                    _mhp["C"]        = _c1.number_input("C", 0.001, 1000.0, 1.0, key=f"ch_lr_C_{_mn}")
                    _mhp["max_iter"] = int(_c2.number_input("max_iter", 100, 10000, 1000, key=f"ch_lr_mi_{_mn}"))
                    _mhp["solver"]   = st.selectbox("Solver", ["lbfgs","liblinear","sag","saga"],
                                                     key=f"ch_lr_s_{_mn}")
                elif _mn == "Random Forest":
                    _c1, _c2 = st.columns(2)
                    _mhp["n_estimators"]    = _c1.slider("n_estimators", 10, 500, 100, key=f"ch_rf_ne_{_mn}")
                    _mhp["max_depth"]       = _c2.slider("max_depth (0=None)", 0, 50, 0, key=f"ch_rf_md_{_mn}")
                    _c3, _c4 = st.columns(2)
                    _mhp["min_samples_split"] = _c3.slider("min_samples_split", 2, 20, 2, key=f"ch_rf_mss_{_mn}")
                    _mhp["max_features"]    = _c4.selectbox("max_features", ["sqrt","log2","None"],
                                                              key=f"ch_rf_mf_{_mn}")
                    if _mhp["max_features"] == "None": _mhp["max_features"] = None
                elif _mn == "Decision Tree":
                    _c1, _c2 = st.columns(2)
                    _mhp["max_depth"]         = _c1.slider("max_depth (0=None)", 0, 50, 0, key=f"ch_dt_md_{_mn}")
                    _mhp["min_samples_split"] = _c2.slider("min_samples_split", 2, 20, 2, key=f"ch_dt_mss_{_mn}")
                    _mhp["criterion"]         = st.selectbox("criterion", ["gini","entropy"], key=f"ch_dt_cr_{_mn}")
                elif _mn == "SVM":
                    _c1, _c2 = st.columns(2)
                    _mhp["C"]      = _c1.number_input("C", 0.001, 1000.0, 1.0, key=f"ch_svm_C_{_mn}")
                    _mhp["kernel"] = _c2.selectbox("Kernel", ["rbf","linear","poly"], key=f"ch_svm_k_{_mn}")
                    _mhp["gamma"]  = st.selectbox("Gamma", ["scale","auto"], key=f"ch_svm_g_{_mn}")
                elif _mn == "K-Nearest Neighbors":
                    _c1, _c2 = st.columns(2)
                    _mhp["n_neighbors"] = _c1.slider("n_neighbors", 1, 50, 5, key=f"ch_knn_n_{_mn}")
                    _mhp["weights"]     = _c2.selectbox("Weights", ["uniform","distance"], key=f"ch_knn_w_{_mn}")
                    _mhp["metric"]      = st.selectbox("Metric",
                        ["minkowski","euclidean","manhattan"], key=f"ch_knn_m_{_mn}")
                elif _mn == "Naive Bayes":
                    _mhp["var_smoothing"] = st.number_input("var_smoothing", 1e-12, 1e-3, 1e-9,
                                                             format="%.2e", key=f"ch_nb_vs_{_mn}")
                elif _mn == "Gradient Boosting":
                    _c1, _c2 = st.columns(2)
                    _mhp["n_estimators"]  = _c1.slider("n_estimators", 50, 500, 100, key=f"ch_gb_ne_{_mn}")
                    _mhp["learning_rate"] = _c2.slider("learning_rate", 0.01, 1.0, 0.1, key=f"ch_gb_lr_{_mn}")
                    _mhp["max_depth"]     = st.slider("max_depth", 1, 15, 3, key=f"ch_gb_md_{_mn}")
                elif _mn == "XGBoost":
                    _c1, _c2 = st.columns(2)
                    _mhp["n_estimators"]  = _c1.slider("n_estimators", 50, 500, 100, key=f"ch_xgb_ne_{_mn}")
                    _mhp["learning_rate"] = _c2.slider("learning_rate", 0.01, 1.0, 0.1, key=f"ch_xgb_lr_{_mn}")
                    _c3, _c4 = st.columns(2)
                    _mhp["max_depth"]  = _c3.slider("max_depth", 1, 15, 6, key=f"ch_xgb_md_{_mn}")
                    _mhp["subsample"]  = _c4.slider("subsample", 0.4, 1.0, 1.0, step=0.05, key=f"ch_xgb_ss_{_mn}")
                _churn_hp[_mn] = _mhp
    st.session_state["churn_hp"] = _churn_hp

    _ch_cv   = st.slider("K-Folds (CV):", 3, 10, 5, key="churn_cv_sl")
    _ch_bal  = st.selectbox("Balanceo de clases:",
                             ["none","smote","undersample","combined"],
                             format_func=lambda x: {"none":"Sin balanceo","smote":"SMOTE",
                                 "undersample":"Under-sampling","combined":"Híbrido"}[x],
                             key="churn_bal_sel")
    _ch_thr  = st.slider("Umbral de decisión:", 0.1, 0.9, 0.5, 0.05, key="churn_thr_sl")

    if st.button("🚀 Ejecutar Benchmarking de Clasificación", type="primary",
                  use_container_width=True, key="btn_churn_clf"):
        with st.spinner("⏳ Entrenando clasificadores..."):
            try:
                from mlbenchmark.benchmarking import run_benchmark

                _feat_cols = [c for c in _df_proc.columns if c != _target_churn]
                _X = _df_proc[_feat_cols].fillna(0).values.astype(float)
                _y = _df_proc[_target_churn].values.astype(int)

                _models = build_models_with_hyperparams("classification", _sel_clf, _churn_hp)

                _res = run_benchmark(
                    problem_type="classification",
                    X=_X, y=_y,
                    models=_models,
                    test_size=0.3,
                    cv_folds=_ch_cv,
                    threshold=_ch_thr,
                    balancing_technique=_ch_bal,
                    scale=True,
                    random_state=42,
                )

                _clf_results = _res["results"]
                st.session_state["churn_clf_results"] = _clf_results
                st.session_state["churn_X_tr"] = _res["X_train"]
                st.session_state["churn_X_te"] = _res["X_test"]
                st.session_state["churn_y_tr"] = _res["y_train"]
                st.session_state["churn_y_te"] = _res["y_test"]
                st.session_state["churn_models"] = _models
                st.session_state.churn_clf_run = True
                st.success("✅ Benchmarking completado.")
            except Exception as _e:
                import traceback
                st.error(f"❌ {_e}")
                st.code(traceback.format_exc())

    if st.session_state.churn_clf_run and "churn_clf_results" in st.session_state:
        _res = st.session_state["churn_clf_results"]
        if isinstance(_res, pd.DataFrame) and not _res.empty:
            st.markdown('<span class="phase-badge ph5">Fase 5 · Evaluation</span>', unsafe_allow_html=True)
            st.subheader("📊 Resultados de Clasificación")
            show_df(style_table(_res))
            _best_row = _res.loc[_res["F1"].idxmax()] if "F1" in _res.columns else _res.iloc[0]
            st.success(f"🥇 Mejor modelo: **{_best_row.name if hasattr(_best_row, 'name') else '—'}** "
                        f"· F1={fmt(_best_row.get('F1', None))} "
                        f"· AUC-ROC={fmt(_best_row.get('AUC-ROC', None))}")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # FASE 4b: K-MEANS CLUSTERING
    # ══════════════════════════════════════════════════════════
    st.markdown('<span class="phase-badge ph4">Fase 4 · Modeling — K-Means Clustering</span>', unsafe_allow_html=True)
    st.subheader("🔵 Segmentación K-Means")

    _km_k_min  = st.slider("k mínimo:", 2, 5,  2, key="km_k_min_sl")
    _km_k_max  = st.slider("k máximo:", 3, 15, 8, key="km_k_max_sl")
    _km_k_opt  = st.slider("k a usar (tras analizar codo/silhouette):", 2, 15, 3, key="km_k_opt_sl")

    if st.button("🔵 Ejecutar K-Means + Análisis Elbow & Silhouette",
                  use_container_width=True, key="btn_churn_km"):
        with st.spinner("⏳ Calculando clusters..."):
            try:
                from mlbenchmark.clustering import KMeansClusterer, cluster_churn_profile
                from sklearn.preprocessing import StandardScaler

                _feat_cols_km = [c for c in _df_proc.columns if c != _target_churn]
                _X_km = _df_proc[_feat_cols_km].fillna(0).values.astype(float)
                _X_km = StandardScaler().fit_transform(_X_km)

                _clusterer = KMeansClusterer(random_state=42)
                _k_range = range(_km_k_min, _km_k_max + 1)
                _df_elbow, _df_sil = _clusterer.run_analysis(_X_km, _k_range)
                _clusterer.fit(_X_km, k=_km_k_opt)

                _df_pca = _clusterer.pca_2d(_X_km)
                _df_profile = _clusterer.get_cluster_profile(_df_proc[_feat_cols_km + [_target_churn]])

                # Churn rate por cluster
                _df_with_cluster = _df_proc.copy()
                _df_with_cluster["Cluster"] = _clusterer.labels_
                _df_churn_rate = cluster_churn_profile(_df_with_cluster, "Cluster", _target_churn)

                st.session_state["churn_km_elbow"]    = _df_elbow
                st.session_state["churn_km_sil"]      = _df_sil
                st.session_state["churn_km_pca"]      = _df_pca
                st.session_state["churn_km_profile"]  = _df_profile
                st.session_state["churn_km_churnrate"]= _df_churn_rate
                st.session_state.churn_km_run = True
                st.success(f"✅ {_km_k_opt} clusters generados · "
                            f"Inercia: {_clusterer.inertia_:,.1f}")
            except Exception as _e:
                import traceback
                st.error(f"❌ {_e}")
                st.code(traceback.format_exc())

    if st.session_state.churn_km_run:
        _c_el, _c_si = st.columns(2)
        with _c_el:
            _df_elbow = st.session_state["churn_km_elbow"]
            fig_elbow = px.line(_df_elbow, x="k", y="wcss", markers=True,
                                 title="Método del Codo (WCSS)", template=TMPL)
            fig_elbow.update_traces(line_color=DISC[0])
            st.plotly_chart(fig_elbow, width="stretch")
        with _c_si:
            _df_sil = st.session_state["churn_km_sil"]
            fig_sil = px.line(_df_sil, x="k", y="silhouette", markers=True,
                               title="Silhouette Score por k", template=TMPL)
            fig_sil.update_traces(line_color=DISC[1])
            st.plotly_chart(fig_sil, width="stretch")

        _df_pca = st.session_state["churn_km_pca"]
        fig_pca = px.scatter(_df_pca, x="PC1", y="PC2", color="Cluster",
                              title=f"Visualización PCA 2D — k={_km_k_opt} clusters",
                              template=TMPL, color_discrete_sequence=DISC)
        st.plotly_chart(fig_pca, width="stretch")

        st.subheader("📋 Perfil de Clusters")
        show_df(style_table(st.session_state["churn_km_profile"].reset_index()))

        st.subheader("📉 Tasa de Churn por Cluster")
        show_df(st.session_state["churn_km_churnrate"])

    st.divider()

    # ══════════════════════════════════════════════════════════
    # FASE 4c: REGLAS DE ASOCIACIÓN — CHURN-SPECIFIC
    # ══════════════════════════════════════════════════════════
    st.markdown('<span class="phase-badge ph4">Fase 4 · Modeling — Reglas de Asociación</span>', unsafe_allow_html=True)
    st.subheader("🔗 Reglas de Asociación orientadas a Churn")

    _ar_sup_ch  = st.slider("Soporte mínimo:", 0.005, 0.20, 0.02, 0.005, format="%.3f", key="ch_ar_sup")
    _ar_conf_ch = st.slider("Confianza mínima:", 0.05, 0.90, 0.30, 0.05, key="ch_ar_conf")
    _ar_lift_ch = st.slider("Lift mínimo:", 1.0, 5.0, 1.2, 0.1, key="ch_ar_lift")
    _ar_bins_ch = st.slider("Bins para discretización numérica:", 2, 6, 4, key="ch_ar_bins")

    if st.button("🚀 Ejecutar Reglas de Asociación (Churn)", use_container_width=True,
                  type="primary", key="btn_churn_ar"):
        with st.spinner("⏳ Minando reglas..."):
            try:
                from mlbenchmark.association_rules import AssociationRulesMiner

                _ar_miner_ch = AssociationRulesMiner()
                _ar_miner_ch.fit_from_churn_dataframe(
                    _df_raw,
                    churn_col=_target_churn if _target_churn in _df_raw.columns else "Churn",
                    numeric_bins=_ar_bins_ch,
                )
                _df_ch_rules = _ar_miner_ch.get_churn_rules(
                    min_confidence=_ar_conf_ch,
                    min_lift=_ar_lift_ch,
                    min_support=_ar_sup_ch,
                )
                _df_all_rules_ch = _ar_miner_ch.get_rules(
                    min_confidence=_ar_conf_ch,
                    min_lift=_ar_lift_ch,
                    min_support=_ar_sup_ch,
                )

                st.session_state["churn_ar_miner"]      = _ar_miner_ch
                st.session_state["churn_ar_rules"]      = _df_ch_rules
                st.session_state["churn_ar_all_rules"]  = _df_all_rules_ch
                st.session_state.churn_ar_run = True

                _n_all = len(_df_all_rules_ch)
                _n_ch  = len(_df_ch_rules)
                st.success(f"✅ {_n_all} reglas totales · **{_n_ch} reglas con consecuente Churn=Yes**")
            except Exception as _e:
                import traceback
                st.error(f"❌ {_e}")
                st.code(traceback.format_exc())

    if st.session_state.churn_ar_run and "churn_ar_rules" in st.session_state:
        _df_ch_r = st.session_state["churn_ar_rules"]
        if not _df_ch_r.empty:
            st.subheader("📋 Top Reglas → Churn=Yes")
            _top_ch = _df_ch_r[["antecedents_str","consequents_str","support","confidence","lift"]].head(20)
            show_df(_top_ch)

            fig_ch_ar = px.bar(
                _df_ch_r.head(15), x="lift", y="antecedents_str",
                orientation="h", color="confidence",
                color_continuous_scale=C_REV,
                title="Top 15 Reglas de Churn por Lift",
                template=TMPL, labels={"antecedents_str": "Antecedente", "lift": "Lift"},
            )
            st.plotly_chart(fig_ch_ar, width="stretch")
        else:
            st.warning("No se generaron reglas con consecuente Churn=Yes. "
                       "Reduce el soporte, confianza o lift mínimo.")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # FASE 4d: REDES NEURONALES
    # ══════════════════════════════════════════════════════════
    st.markdown('<span class="phase-badge ph4">Fase 4 · Modeling — Redes Neuronales</span>', unsafe_allow_html=True)
    st.subheader("🧠 Benchmarking de Redes Neuronales")

    _nn_epochs_ch = st.slider("Épocas (Keras):", 10, 100, 30, 5, key="ch_nn_epochs")
    _nn_test_ch   = st.slider("Test size (%):", 10, 40, 20, 5, key="ch_nn_test") / 100

    if st.button("🚀 Entrenar Redes Neuronales sobre Churn", use_container_width=True,
                  type="primary", key="btn_churn_nn"):
        with st.spinner("⏳ Entrenando redes... (puede tomar 1-5 min)"):
            try:
                from mlbenchmark.neural_networks import benchmark_neural_networks
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                _feat_cols_nn = [c for c in _df_proc.columns if c != _target_churn]
                _X_nn = _df_proc[_feat_cols_nn].fillna(0).values.astype(float)
                _y_nn = _df_proc[_target_churn].values.astype(int)
                _X_nn = StandardScaler().fit_transform(_X_nn)

                _X_tr_nn, _X_te_nn, _y_tr_nn, _y_te_nn = train_test_split(
                    _X_nn, _y_nn, test_size=_nn_test_ch, random_state=42, stratify=_y_nn)

                _df_nn_res = benchmark_neural_networks(
                    _X_tr_nn, _X_te_nn, _y_tr_nn, _y_te_nn,
                    task="classification", epochs=_nn_epochs_ch)

                st.session_state["churn_nn_results"] = _df_nn_res
                st.session_state.churn_nn_run = True
                st.success("✅ Redes neuronales entrenadas.")
            except Exception as _e:
                import traceback
                st.error(f"❌ {_e}")
                st.code(traceback.format_exc())

    if st.session_state.churn_nn_run and "churn_nn_results" in st.session_state:
        _df_nn_r = st.session_state["churn_nn_results"]
        if isinstance(_df_nn_r, pd.DataFrame) and not _df_nn_r.empty:
            st.subheader("📊 Resultados de Redes Neuronales")
            show_df(style_table(_df_nn_r))

    st.divider()

    # ══════════════════════════════════════════════════════════
    # FASE 5: EVALUACIÓN ESTADÍSTICA — Prueba de Wilcoxon
    # ══════════════════════════════════════════════════════════
    st.markdown('<span class="phase-badge ph5">Fase 5 · Evaluation — Prueba Estadística</span>', unsafe_allow_html=True)
    st.subheader("📐 Prueba de Wilcoxon (Signed-Rank Test)")
    st.markdown("""
    Compara las distribuciones de F1-Score de **dos modelos** usando validación cruzada
    repetida (RepeatedStratifiedKFold). Si p < 0.05, la diferencia es estadísticamente significativa.
    """)

    _wil_cv_n   = st.slider("Repeticiones CV:", 2, 10, 5, key="wil_cv_n_sl")
    _wil_cv_k   = st.slider("K folds:", 3, 10, 5,  key="wil_cv_k_sl")

    if st.button("📐 Ejecutar Prueba de Wilcoxon (top 2 modelos)", use_container_width=True,
                  key="btn_churn_wilcoxon"):
        if not st.session_state.churn_clf_run or "churn_models" not in st.session_state:
            st.error("Primero ejecuta el benchmarking de clasificación.")
        else:
            with st.spinner("⏳ Ejecutando CV repetida..."):
                try:
                    from scipy.stats import wilcoxon
                    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
                    from sklearn.preprocessing import StandardScaler

                    _feat_cols_wil = [c for c in _df_proc.columns if c != _target_churn]
                    _X_wil = _df_proc[_feat_cols_wil].fillna(0).values.astype(float)
                    _y_wil = _df_proc[_target_churn].values.astype(int)
                    _X_wil = StandardScaler().fit_transform(_X_wil)

                    _models_wil = st.session_state["churn_models"]
                    _rskf = RepeatedStratifiedKFold(n_splits=_wil_cv_k,
                                                    n_repeats=_wil_cv_n, random_state=42)

                    _cv_scores = {}
                    for _mn, _m in _models_wil.items():
                        _scores = cross_val_score(_m, _X_wil, _y_wil, cv=_rskf,
                                                   scoring="f1", n_jobs=-1)
                        _cv_scores[_mn] = _scores

                    # Prueba Wilcoxon entre el par con mayor diferencia de F1 media
                    _means = {k: v.mean() for k, v in _cv_scores.items()}
                    _sorted_models = sorted(_means, key=_means.get, reverse=True)
                    if len(_sorted_models) >= 2:
                        _m1, _m2 = _sorted_models[0], _sorted_models[1]
                        _s1, _s2 = _cv_scores[_m1], _cv_scores[_m2]
                        _stat, _pval = wilcoxon(_s1, _s2)
                    else:
                        _m1 = _sorted_models[0]; _m2 = "—"
                        _stat, _pval = None, None

                    st.session_state["churn_wil_scores"] = _cv_scores
                    st.session_state["churn_wil_pair"]   = (_m1, _m2)
                    st.session_state["churn_wil_stat"]   = (_stat, _pval)
                    st.session_state.churn_stat_run = True
                    st.success("✅ Prueba de Wilcoxon completada.")
                except Exception as _e:
                    import traceback
                    st.error(f"❌ {_e}")
                    st.code(traceback.format_exc())

    if st.session_state.churn_stat_run and "churn_wil_scores" in st.session_state:
        _cv_sc  = st.session_state["churn_wil_scores"]
        _pair   = st.session_state["churn_wil_pair"]
        _wstat  = st.session_state["churn_wil_stat"]

        st.subheader("📊 Distribución de F1-Score por CV Repetida")
        _df_box = pd.DataFrame({k: v for k, v in _cv_sc.items()})
        fig_box = px.box(_df_box, template=TMPL, color_discrete_sequence=DISC,
                          title=f"F1-Score — {_wil_cv_n}×{_wil_cv_k}-Fold CV")
        st.plotly_chart(fig_box, width="stretch")

        if _wstat[0] is not None:
            st.markdown(f"""
            **Prueba Wilcoxon Signed-Rank:**
            - Modelos comparados: **{_pair[0]}** vs **{_pair[1]}**
            - Estadístico W: `{_wstat[0]:.4f}`
            - p-valor: `{_wstat[1]:.4f}`
            - Interpretación: {"✅ **Diferencia estadísticamente significativa** (p < 0.05)"
                if _wstat[1] < 0.05
                else "⚠️ No hay diferencia significativa (p ≥ 0.05)"}
            """)

        # Tabla resumen CV
        _df_cv_sum = pd.DataFrame({
            "Modelo": list(_cv_sc.keys()),
            "F1 media": [v.mean().round(4) for v in _cv_sc.values()],
            "F1 std":   [v.std().round(4)  for v in _cv_sc.values()],
            "F1 min":   [v.min().round(4)  for v in _cv_sc.values()],
            "F1 max":   [v.max().round(4)  for v in _cv_sc.values()],
        }).sort_values("F1 media", ascending=False)
        show_df(style_table(_df_cv_sum))

    st.divider()
    st.markdown('<span class="phase-badge ph6">Fase 6 · Deployment</span>', unsafe_allow_html=True)
    st.info("""
    **Para el artículo científico:** exporta los resultados de cada fase como tablas LaTeX
    (usa `df.to_latex()` en Python) e insértalas en la sección *Resultados y Discusión*.
    Las visualizaciones pueden descargarse como PNG desde los menús de Plotly.
    """)


# ╔══════════════════════════════════════════════════════════╗
# ║  DESCARGA GLOBAL DE RESULTADOS — CSV                    ║
# ╚══════════════════════════════════════════════════════════╝
st.divider()
st.subheader("⬇️ Descargar Resultados")
st.caption("Consolida todos los resultados generados en la sesión y descárgalos como CSV.")


def _build_download_csv() -> tuple[str, str]:
    """
    Recolecta todos los DataFrames de resultados del session_state,
    agrega una columna 'Sección' y los concatena verticalmente.

    Returns:
        (csv_string, filename)
    """
    import io
    sections = []

    # ── Benchmarking clásico (Tab 3) ─────────────────────────
    _res_main = st.session_state.get("results")
    if _res_main and isinstance(_res_main, dict):
        _df_main = _res_main.get("results")
        if isinstance(_df_main, pd.DataFrame) and not _df_main.empty:
            _tmp = _df_main[[c for c in _df_main.columns if not c.startswith("_")]].copy()
            _tmp.insert(0, "Sección", "Benchmarking General")
            _tmp.insert(1, "Tipo", _res_main.get("problem_type", "—"))
            sections.append(_tmp)

    # ── Redes Neuronales (Tab 7) ──────────────────────────────
    _df_nn = st.session_state.get("_nn_results")
    if isinstance(_df_nn, pd.DataFrame) and not _df_nn.empty:
        _tmp = _df_nn.copy()
        _tmp.insert(0, "Sección", "Redes Neuronales")
        _tmp.insert(1, "Tipo", "Neural Networks")
        sections.append(_tmp)

    # ── Reglas de Asociación — General (Tab 8) ───────────────
    _df_ar = st.session_state.get("_ar_rules")
    if isinstance(_df_ar, pd.DataFrame) and not _df_ar.empty:
        _keep = ["antecedents_str", "consequents_str", "support", "confidence", "lift", "conviction"]
        _tmp = _df_ar[[c for c in _keep if c in _df_ar.columns]].copy()
        _tmp.insert(0, "Sección", "Reglas de Asociación")
        _tmp.insert(1, "Tipo", "Association Rules")
        sections.append(_tmp)

    # ── Churn: Clasificación (Tab 9) ─────────────────────────
    _res_clf = st.session_state.get("churn_clf_results")
    if isinstance(_res_clf, pd.DataFrame) and not _res_clf.empty:
        _tmp = _res_clf.copy()
        _tmp.insert(0, "Sección", "Churn – Clasificación")
        _tmp.insert(1, "Tipo", "Classification")
        sections.append(_tmp)

    # ── Churn: K-Means perfil (Tab 9) ────────────────────────
    _df_km = st.session_state.get("churn_km_churnrate")
    if isinstance(_df_km, pd.DataFrame) and not _df_km.empty:
        _tmp = _df_km.copy()
        _tmp.insert(0, "Sección", "Churn – K-Means (tasa churn)")
        _tmp.insert(1, "Tipo", "Clustering")
        sections.append(_tmp)

    _df_km_prof = st.session_state.get("churn_km_profile")
    if isinstance(_df_km_prof, pd.DataFrame) and not _df_km_prof.empty:
        _tmp = _df_km_prof.reset_index().copy()
        _tmp.insert(0, "Sección", "Churn – K-Means (perfil)")
        _tmp.insert(1, "Tipo", "Clustering")
        sections.append(_tmp)

    # ── Churn: Reglas de Asociación → Churn=Yes (Tab 9) ──────
    _df_ch_ar = st.session_state.get("churn_ar_rules")
    if isinstance(_df_ch_ar, pd.DataFrame) and not _df_ch_ar.empty:
        _keep = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
        _tmp = _df_ch_ar[[c for c in _keep if c in _df_ch_ar.columns]].copy()
        _tmp.insert(0, "Sección", "Churn – Reglas Asociación (Churn=Yes)")
        _tmp.insert(1, "Tipo", "Association Rules")
        sections.append(_tmp)

    # ── Churn: Redes Neuronales (Tab 9) ──────────────────────
    _df_ch_nn = st.session_state.get("churn_nn_results")
    if isinstance(_df_ch_nn, pd.DataFrame) and not _df_ch_nn.empty:
        _tmp = _df_ch_nn.copy()
        _tmp.insert(0, "Sección", "Churn – Redes Neuronales")
        _tmp.insert(1, "Tipo", "Neural Networks")
        sections.append(_tmp)

    # ── Churn: CV Wilcoxon (Tab 9) ───────────────────────────
    _cv_wil = st.session_state.get("churn_wil_scores")
    if _cv_wil:
        _tmp = pd.DataFrame({
            "Modelo":    list(_cv_wil.keys()),
            "F1_media":  [v.mean().round(4) for v in _cv_wil.values()],
            "F1_std":    [v.std().round(4)  for v in _cv_wil.values()],
            "F1_min":    [v.min().round(4)  for v in _cv_wil.values()],
            "F1_max":    [v.max().round(4)  for v in _cv_wil.values()],
        }).sort_values("F1_media", ascending=False)
        _wstat = st.session_state.get("churn_wil_stat", (None, None))
        _pair  = st.session_state.get("churn_wil_pair",  ("—", "—"))
        _tmp.insert(0, "Sección", "Churn – Wilcoxon CV")
        _tmp.insert(1, "Tipo", "Statistical Test")
        _tmp["Wilcoxon_W"]   = _wstat[0]
        _tmp["Wilcoxon_p"]   = _wstat[1]
        _tmp["Par_modelo_1"] = _pair[0]
        _tmp["Par_modelo_2"] = _pair[1]
        sections.append(_tmp)

    if not sections:
        return None, None

    _df_all = pd.concat(sections, ignore_index=True, sort=False)

    buf = io.StringIO()
    _df_all.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue(), "ml_benchmarking_resultados.csv"


_csv_data, _csv_name = _build_download_csv()

if _csv_data:
    st.download_button(
        label="📥 Descargar todos los resultados como CSV",
        data=_csv_data,
        file_name=_csv_name,
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )
    st.caption(
        "El archivo incluye resultados de: Benchmarking General · Redes Neuronales · "
        "Reglas de Asociación · Churn (Clasificación, K-Means, Reglas, Redes Neuronales, Wilcoxon)."
    )
else:
    st.info("Ejecuta al menos un análisis para habilitar la descarga.")
