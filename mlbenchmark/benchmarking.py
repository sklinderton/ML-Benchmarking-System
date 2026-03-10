"""
benchmarking.py - Orquestador principal del sistema de benchmarking
BCD-7213 Minería de Datos Avanzada - Universidad LEAD

Cambios respecto a la versión anterior:
  - _encode_categoricals(): codifica columnas object/category/bool con LabelEncoder
  - _encode_target(): codifica targets string antes del pipeline
  - _to_numpy(): acepta tanto DataFrames como arrays numpy
  - SMOTE safe_k: ajusta k_neighbors automáticamente según tamaño de clase minoritaria
  - run_benchmark() acepta X, y como DataFrame o ndarray indistintamente
"""

import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder

from .preprocessing import split_data, scale_features
from .balancing import apply_balancing
from .metrics import classification_metrics, regression_metrics, timeseries_metrics
from .validation import stratified_kfold, kfold_validation
from .threshold import apply_threshold
from .models_classification import get_classification_models, predict_classification
from .models_regression import get_regression_models
from .models_timeseries import get_timeseries_models


# ─────────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ─────────────────────────────────────────────────────────────────

def _to_numpy(arr):
    """Convierte DataFrame o Series a ndarray float64 de forma segura."""
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        return arr.values
    return np.asarray(arr)


def _encode_categoricals(X_train, X_test):
    """
    Codifica con LabelEncoder todas las columnas object/category/bool.
    Valores desconocidos en test se marcan como -1.

    Retorna arrays float64.
    """
    # Si ya son ndarrays, convertir a DataFrame temporal para detectar dtypes
    if isinstance(X_train, np.ndarray):
        return X_train.astype(float), X_test.astype(float)

    X_tr = X_train.copy()
    X_te = X_test.copy()

    for col in X_tr.columns:
        if X_tr[col].dtype in (object,) or str(X_tr[col].dtype) in ("category", "bool"):
            le = LabelEncoder()
            X_tr[col] = le.fit_transform(X_tr[col].astype(str))
            # Mapear test; valores fuera de clases → -1
            mapping = {c: i for i, c in enumerate(le.classes_)}
            X_te[col] = X_te[col].astype(str).map(mapping).fillna(-1).astype(int)

    return X_tr.values.astype(float), X_te.values.astype(float)


def _encode_target(y_train, y_test):
    """
    Codifica targets de texto con LabelEncoder.
    Si son numéricos, solo convierte a array.
    """
    y_tr = _to_numpy(y_train)
    y_te = _to_numpy(y_test)

    if y_tr.dtype.kind in ("U", "S", "O"):  # strings / object
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr.astype(str))
        y_te_enc = np.array([le.transform([str(v)])[0]
                              if str(v) in le.classes_ else -1 for v in y_te])
        return y_tr, y_te_enc
    return y_tr.astype(float), y_te.astype(float)


# ─────────────────────────────────────────────────────────────────
# CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────

def benchmark_classification(
    X_train, X_test, y_train, y_test,
    models=None,
    threshold=0.5,
    cv_folds=5,
    balancing_technique="none",
    random_state=42,
):
    """
    Benchmarking completo de modelos de clasificación.

    Returns:
        pd.DataFrame con métricas ordenadas por AUC-ROC
    """
    if models is None:
        models = get_classification_models(random_state)

    # ── Encoding ──────────────────────────────────────────────────
    y_train, y_test = _encode_target(y_train, y_test)
    X_train, X_test = _encode_categoricals(X_train, X_test)

    # ── Balanceo ──────────────────────────────────────────────────
    X_tr_bal, y_tr_bal = apply_balancing(X_train, y_train,
                                         balancing_technique, random_state)

    results = []

    for name, model in models.items():
        try:
            m = copy.deepcopy(model)
            m.fit(X_tr_bal, y_tr_bal)

            y_pred, y_prob = predict_classification(m, X_test, threshold)
            met = classification_metrics(y_test, y_pred, y_prob, threshold)

            m_cv = copy.deepcopy(model)
            cv_res = stratified_kfold(m_cv, X_train, y_train,
                                      k=cv_folds, scoring="roc_auc",
                                      random_state=random_state)

            results.append({
                "Model":            name,
                "Accuracy":         met["accuracy"],
                "Precision":        met["precision"],
                "Recall":           met["recall"],
                "F1-Score":         met["f1_score"],
                "AUC-ROC":          met["auc_roc"] if met["auc_roc"] else 0.5,
                "CV Mean":          cv_res["mean"],
                "CV Std":           cv_res["std"],
                "CV Scores":        cv_res["scores"],
                "_y_prob":          y_prob,
                "_confusion_matrix": met["confusion_matrix"],
            })

        except Exception as e:
            results.append({
                "Model": name,
                "Accuracy": None, "Precision": None, "Recall": None,
                "F1-Score": None, "AUC-ROC": None,
                "CV Mean": None, "CV Std": None, "CV Scores": [],
                "_y_prob": None, "_confusion_matrix": None,
                "_error": str(e),
            })

    df = pd.DataFrame(results)
    df = df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# REGRESIÓN
# ─────────────────────────────────────────────────────────────────

def benchmark_regression(
    X_train, X_test, y_train, y_test,
    models=None,
    cv_folds=5,
    random_state=42,
):
    """
    Benchmarking completo de modelos de regresión.

    Returns:
        pd.DataFrame con métricas ordenadas por R²
    """
    if models is None:
        models = get_regression_models(random_state)

    # ── Encoding ──────────────────────────────────────────────────
    y_train, y_test = _encode_target(y_train, y_test)
    X_train, X_test = _encode_categoricals(X_train, X_test)

    results = []

    for name, model in models.items():
        try:
            m = copy.deepcopy(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            met = regression_metrics(y_test, y_pred)

            m_cv = copy.deepcopy(model)
            cv_res = kfold_validation(m_cv, X_train, y_train,
                                      k=cv_folds, scoring="r2",
                                      random_state=random_state)

            results.append({
                "Model":        name,
                "MSE":          met["mse"],
                "RMSE":         met["rmse"],
                "MAE":          met["mae"],
                "R²":           met["r2"],
                "CV Mean (R²)": cv_res["mean"],
                "CV Std":       cv_res["std"],
                "CV Scores":    cv_res["scores"],
            })

        except Exception as e:
            results.append({
                "Model": name,
                "MSE": None, "RMSE": None, "MAE": None,
                "R²": None, "CV Mean (R²)": None, "CV Std": None,
                "CV Scores": [], "_error": str(e),
            })

    df = pd.DataFrame(results)
    df = df.sort_values("R²", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# SERIES DE TIEMPO
# ─────────────────────────────────────────────────────────────────

def benchmark_timeseries(
    train, test,
    models=None,
    seasonal_periods=12,
):
    """
    Benchmarking de modelos de series de tiempo.

    Returns:
        pd.DataFrame con métricas ordenadas por RMSE (ascendente)
    """
    if models is None:
        models = get_timeseries_models(seasonal_periods)

    steps = len(test)
    results = []

    for name, model in models.items():
        try:
            m = copy.deepcopy(model)
            m.fit(train)
            preds = np.array(m.predict(steps))

            met = timeseries_metrics(test, preds)

            results.append({
                "Model":       name,
                "MSE":         met["mse"],
                "RMSE":        met["rmse"],
                "MAE":         met["mae"],
                "MAPE (%)":    met["mape"],
                "_predictions": preds.tolist(),
            })

        except Exception as e:
            results.append({
                "Model": name,
                "MSE": None, "RMSE": None, "MAE": None, "MAPE (%)": None,
                "_predictions": None, "_error": str(e),
            })

    df = pd.DataFrame(results)
    df_ok  = df.dropna(subset=["RMSE"]).sort_values("RMSE")
    df_err = df[df["RMSE"].isna()]
    return pd.concat([df_ok, df_err]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN UNIFICADA
# ─────────────────────────────────────────────────────────────────

def run_benchmark(
    problem_type,
    X=None, y=None,
    series=None,
    models=None,
    test_size=0.3,
    cv_folds=5,
    threshold=0.5,
    balancing_technique="none",
    scale=True,
    seasonal_periods=12,
    train_ratio=0.8,
    random_state=42,
):
    """
    Función unificada para ejecutar benchmarking.

    Acepta X, y como DataFrame, Series o ndarray indistintamente.
    Maneja encoding de categóricas, escalado y balanceo automáticamente.

    Args:
        problem_type : 'classification' | 'regression' | 'timeseries'
        X, y         : Features y target (DataFrame o ndarray)
        series       : Array para series de tiempo
        models       : dict {nombre: instancia_sklearn}  (None = defaults)
        test_size    : Proporción test (0.1–0.5)
        cv_folds     : Número de folds K-Fold
        threshold    : Umbral de decisión (solo clasificación)
        balancing_technique : 'none'|'smote'|'undersample'|'combined'
        scale        : Aplicar StandardScaler
        seasonal_periods : Períodos estacionales (series de tiempo)
        train_ratio  : Ratio train/test (series de tiempo)
        random_state : Semilla aleatoria

    Returns:
        dict con 'results' DataFrame y datos de contexto
    """
    problem_type = problem_type.lower()

    if problem_type in ("classification", "regression"):
        # Convertir a numpy si son DataFrames, pero conservar info de dtypes
        # para el encoding (pasamos directamente los DFs al split y luego encode)
        X_arr = X if isinstance(X, pd.DataFrame) else _to_numpy(X)
        y_arr = _to_numpy(y)

        stratify = (problem_type == "classification")
        X_train, X_test, y_train, y_test = split_data(
            X_arr, y_arr,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        if scale:
            # Encoding previo al scaling (necesita arrays float)
            y_train_enc, y_test_enc = _encode_target(y_train, y_test)
            X_train_enc, X_test_enc = _encode_categoricals(X_train, X_test)
            X_train_s, X_test_s, scaler = scale_features(X_train_enc, X_test_enc)
            # Pasar ya codificados y escalados, encoding dentro de benchmark_* no hará nada extra
            X_train_final, X_test_final = X_train_s, X_test_s
            y_train_final, y_test_final = y_train_enc, y_test_enc
        else:
            X_train_final, X_test_final = X_train, X_test
            y_train_final, y_test_final = y_train, y_test
            scaler = None

        if problem_type == "classification":
            df = benchmark_classification(
                X_train_final, X_test_final,
                y_train_final, y_test_final,
                models=models,
                threshold=threshold,
                cv_folds=cv_folds,
                balancing_technique=balancing_technique,
                random_state=random_state,
            )
            return {
                "results":   df,
                "X_train":   X_train_final,
                "X_test":    X_test_final,
                "y_train":   y_train_final,
                "y_test":    y_test_final,
                "scaler":    scaler,
                "problem_type": problem_type,
            }
        else:
            df = benchmark_regression(
                X_train_final, X_test_final,
                y_train_final, y_test_final,
                models=models,
                cv_folds=cv_folds,
                random_state=random_state,
            )
            return {
                "results":   df,
                "X_train":   X_train_final,
                "X_test":    X_test_final,
                "y_train":   y_train_final,
                "y_test":    y_test_final,
                "scaler":    scaler,
                "problem_type": problem_type,
            }

    elif problem_type == "timeseries":
        if isinstance(series, pd.Series):
            series = series.values.astype(float)
        series = np.asarray(series, dtype=float)

        split_idx   = int(len(series) * train_ratio)
        train, test = series[:split_idx], series[split_idx:]

        df = benchmark_timeseries(
            train, test,
            models=models,
            seasonal_periods=seasonal_periods,
        )
        return {
            "results":      df,
            "train":        train,
            "test":         test,
            "problem_type": problem_type,
        }

    else:
        raise ValueError(
            f"Tipo de problema no válido: '{problem_type}'. "
            "Opciones: 'classification', 'regression', 'timeseries'"
        )


def rank_models(df, metric=None, ascending=None):
    """Ordena el DataFrame de resultados por la métrica especificada."""
    if metric is None:
        if "AUC-ROC"  in df.columns: metric, ascending = "AUC-ROC",  False
        elif "R²"     in df.columns: metric, ascending = "R²",       False
        elif "RMSE"   in df.columns: metric, ascending = "RMSE",     True
        else: return df

    if ascending is None:
        ascending = metric in ("RMSE", "MSE", "MAE", "MAPE (%)")

    return df.sort_values(metric, ascending=ascending).reset_index(drop=True)