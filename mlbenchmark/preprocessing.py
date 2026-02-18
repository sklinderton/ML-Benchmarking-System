"""
preprocessing.py - Módulo de preprocesamiento de datos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def split_data(X, y, test_size=0.3, random_state=42, stratify=True):
    """
    Divide los datos en train y test con opción de estratificación.

    Args:
        X: Features
        y: Target
        test_size: Proporción del set de prueba (0.1 - 0.5)
        random_state: Semilla aleatoria
        stratify: Si True, mantiene proporción de clases

    Returns:
        X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    try:
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=strat)
    except ValueError:
        # Si falla estratificación (p.ej. regresión), sin estratificar
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)


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
