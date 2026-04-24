"""
preprocessing.py - Módulo de preprocesamiento de datos

MEJORA: split_data ahora detecta URLs y carga datos automáticamente.
  - Si X es una URL string, descarga el dataset (CSV/JSON/Excel/scraping)
  - y puede ser el nombre de la columna target (string) o None (última columna)
  - Todo lo demás sigue igual: firma pública no cambiada
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def split_data(X, y, test_size=0.3, random_state=42, stratify=True):
    """
    Divide los datos en train y test con opción de estratificación.

    MEJORA: si X es una URL (string comenzando con http/https o terminando en
    .csv/.json/.xlsx), carga automáticamente el dataset. En ese caso:
      - y puede ser el nombre de la columna target (string)
      - y=None usa la última columna como target

    Args:
        X: Features, ndarray, DataFrame, o URL string al dataset
        y: Target, ndarray, Series, nombre de columna (str), o None si X es URL
        test_size: Proporción del set de prueba (0.1 - 0.5)
        random_state: Semilla aleatoria
        stratify: Si True, mantiene proporción de clases

    Returns:
        X_train, X_test, y_train, y_test
    """
    # MEJORA: detectar si X es una URL o ruta a archivo de datos
    if isinstance(X, str):
        X_loaded, y_loaded = _load_from_source(X, target_col=y)
        X = X_loaded
        y = y_loaded

    strat = y if stratify else None
    try:
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=strat)
    except ValueError:
        # Si falla estratificación (p.ej. regresión), sin estratificar
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)


def _load_from_source(source: str, target_col=None):
    """
    MEJORA: función interna que carga X e y desde una URL o archivo local.
    No es parte de la API pública pero complementa split_data.

    Args:
        source    : URL o ruta local al dataset
        target_col: nombre de la columna target (str), o None → última columna

    Returns:
        X (DataFrame), y (Series)
    """
    df = None
    src_lower = source.lower().split("?")[0]

    # ── Intentar carga directa con pandas ─────────────────────
    loaders = [
        # (condición, función)
        (lambda s: s.endswith(".csv") or s.endswith(".tsv"),
         lambda: pd.read_csv(source)),
        (lambda s: s.endswith(".json"),
         lambda: pd.read_json(source)),
        (lambda s: s.endswith(".xlsx") or s.endswith(".xls"),
         lambda: pd.read_excel(source)),
    ]

    for condition, loader in loaders:
        if condition(src_lower):
            try:
                df = loader()
                break
            except Exception:
                pass

    # MEJORA: intentar pandas.read_csv como catch-all (funciona con URLs raw)
    if df is None:
        try:
            df = pd.read_csv(source)
        except Exception:
            pass

    # MEJORA: fallback a tabla HTML si la URL es una página web
    if df is None:
        try:
            tables = pd.read_html(source)
            if tables:
                df = max(tables, key=len)
        except Exception:
            pass

    # MEJORA: fallback a WebMiner para páginas de scraping
    if df is None:
        try:
            from .web_mining import WebMiner
            miner = WebMiner(delay=0.5)
            df, _ = miner.scrape_with_log(url_base=source, max_pages=2)
        except Exception:
            pass

    if df is None or df.empty:
        raise ValueError(
            f"No se pudo cargar el dataset desde '{source}'. "
            "Verifica la URL o el formato del archivo."
        )

    # ── Separar X e y ─────────────────────────────────────────
    if isinstance(target_col, str) and target_col in df.columns:
        # MEJORA: columna target especificada explícitamente
        y_out = df[target_col]
        X_out = df.drop(columns=[target_col])
    elif target_col is None or (isinstance(target_col, str) and target_col not in df.columns):
        # MEJORA: usar última columna como target por defecto
        y_out = df.iloc[:, -1]
        X_out = df.iloc[:, :-1]
    else:
        # target_col es array/Series: usar directamente
        y_out = pd.Series(target_col)
        X_out = df

    # MEJORA: codificar categóricas del X automáticamente para compatibilidad
    for col in X_out.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_out = X_out.copy()
        X_out[col] = le.fit_transform(X_out[col].astype(str))

    return X_out, y_out


def scale_features(X_train, X_test, method="standard"):
    """
    Escala features usando StandardScaler o MinMaxScaler.

    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        method: 'standard' (z-score) o 'minmax' (0-1)

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def encode_categorical(df, columns=None):
    """
    Codifica variables categóricas con LabelEncoder.

    Args:
        df: DataFrame
        columns: Lista de columnas a codificar (None = automático)

    Returns:
        df_encoded, encoders dict
    """
    df_enc = df.copy()
    encoders = {}

    if columns is None:
        columns = df_enc.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    return df_enc, encoders


def impute_missing(df, strategy="median", fill_value=None):
    """
    Imputa valores faltantes en un DataFrame.

    Args:
        df:         DataFrame a imputar (se devuelve copia).
        strategy:   'median' | 'mean' | 'mode' | 'constant'
        fill_value: Valor a usar cuando strategy='constant'.

    Returns:
        df_imputed (DataFrame)
    """
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    cat_cols = df_out.select_dtypes(include=["object", "category"]).columns

    if strategy == "median":
        df_out[num_cols] = df_out[num_cols].fillna(df_out[num_cols].median())
        for c in cat_cols:
            mode = df_out[c].mode()
            df_out[c] = df_out[c].fillna(mode[0] if len(mode) else "Unknown")
    elif strategy == "mean":
        df_out[num_cols] = df_out[num_cols].fillna(df_out[num_cols].mean())
        for c in cat_cols:
            mode = df_out[c].mode()
            df_out[c] = df_out[c].fillna(mode[0] if len(mode) else "Unknown")
    elif strategy == "mode":
        for c in df_out.columns:
            mode = df_out[c].mode()
            df_out[c] = df_out[c].fillna(mode[0] if len(mode) else (0 if c in num_cols else "Unknown"))
    elif strategy == "constant":
        df_out = df_out.fillna(fill_value if fill_value is not None else 0)
    return df_out


def encode_categorical_ohe(df, columns=None, drop_first=True):
    """
    Codifica variables categóricas con One-Hot Encoding (pd.get_dummies).

    Args:
        df:         DataFrame.
        columns:    Columnas a codificar (None = automático).
        drop_first: Si True, elimina la primera categoría para evitar multicolinealidad.

    Returns:
        df_encoded (DataFrame con dummies)
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)


def split_timeseries(series, train_ratio=0.8):
    """
    Divide una serie temporal manteniendo el orden temporal.

    Args:
        series: Array o Serie de tiempo
        train_ratio: Proporción de entrenamiento

    Returns:
        train, test
    """
    n = len(series)
    split_idx = int(n * train_ratio)
    return series[:split_idx], series[split_idx:]


def normalize_timeseries(train, test):
    """
    Normaliza series temporales con Min-Max (ajustado solo en train).

    Returns:
        train_norm, test_norm, (min_val, max_val)
    """
    min_val = np.min(train)
    max_val = np.max(train)
    denom = max_val - min_val if max_val != min_val else 1.0
    train_norm = (train - min_val) / denom
    test_norm = (test - min_val) / denom
    return train_norm, test_norm, (min_val, max_val)


def denormalize_timeseries(data_norm, min_val, max_val):
    """Desnormaliza datos de series temporales."""
    return data_norm * (max_val - min_val) + min_val


def create_sequences(data, window_size=12):
    """
    Crea secuencias de entrada/salida para LSTM.

    Args:
        data: Array normalizado
        window_size: Tamaño de la ventana de tiempo

    Returns:
        X (shape: samples, window, 1), y
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)