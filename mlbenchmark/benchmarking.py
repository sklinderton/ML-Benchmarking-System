"""
benchmarking.py - Orquestador principal del sistema de benchmarking
"""

import numpy as np
import pandas as pd
import copy

from .preprocessing import split_data, scale_features
from .balancing import apply_balancing
from .metrics import classification_metrics, regression_metrics, timeseries_metrics
from .validation import stratified_kfold, kfold_validation
from .threshold import apply_threshold
from .models_classification import get_classification_models, predict_classification
from .models_regression import get_regression_models
from .models_timeseries import get_timeseries_models


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

    Args:
        X_train, X_test, y_train, y_test: Datos divididos
        models: dict de modelos (None = todos por defecto)
        threshold: Umbral de decisión
        cv_folds: Número de folds para K-Fold
        balancing_technique: 'none', 'smote', 'undersample', 'combined'
        random_state: Semilla

    Returns:
        pd.DataFrame con métricas ordenadas por AUC-ROC
    """
    if models is None:
        models = get_classification_models(random_state)

    # Balanceo
    X_tr_bal, y_tr_bal = apply_balancing(X_train, y_train,
                                         balancing_technique, random_state)

    results = []

    for name, model in models.items():
        try:
            # Clonar para no afectar instancias externas
            m = copy.deepcopy(model)
            m.fit(X_tr_bal, y_tr_bal)

            y_pred, y_prob = predict_classification(m, X_test, threshold)

            # Métricas en test
            met = classification_metrics(y_test, y_pred, y_prob, threshold)

            # K-Fold estratificado sobre datos ORIGINALES sin balanceo
            # (más realista para estimar rendimiento real)
            m_cv = copy.deepcopy(model)
            cv_res = stratified_kfold(m_cv, X_train, y_train,
                                      k=cv_folds, scoring="roc_auc",
                                      random_state=random_state)

            results.append({
                "Model": name,
                "Accuracy": met["accuracy"],
                "Precision": met["precision"],
                "Recall": met["recall"],
                "F1-Score": met["f1_score"],
                "AUC-ROC": met["auc_roc"] if met["auc_roc"] else 0.5,
                "CV Mean": cv_res["mean"],
                "CV Std": cv_res["std"],
                "CV Scores": cv_res["scores"],
                # Para curvas ROC / detalles
                "_y_prob": y_prob,
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
                "Model": name,
                "MSE": met["mse"],
                "RMSE": met["rmse"],
                "MAE": met["mae"],
                "R²": met["r2"],
                "CV Mean (R²)": cv_res["mean"],
                "CV Std": cv_res["std"],
                "CV Scores": cv_res["scores"],
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

    Args:
        train: Array de entrenamiento
        test: Array de prueba
        models: dict de modelos (None = todos por defecto)
        seasonal_periods: Períodos estacionales

    Returns:
        pd.DataFrame con métricas ordenadas por RMSE
    """
    if models is None:
        models = get_timeseries_models(seasonal_periods)

    steps = len(test)
    results = []

    for name, model in models.items():
        try:
            m = copy.deepcopy(model)
            m.fit(train)
            preds = m.predict(steps)
            preds = np.array(preds)

            met = timeseries_metrics(test, preds)

            results.append({
                "Model": name,
                "MSE": met["mse"],
                "RMSE": met["rmse"],
                "MAE": met["mae"],
                "MAPE (%)": met["mape"],
                "_predictions": preds.tolist(),
            })

        except Exception as e:
            results.append({
                "Model": name,
                "MSE": None, "RMSE": None, "MAE": None, "MAPE (%)": None,
                "_predictions": None, "_error": str(e),
            })

    df = pd.DataFrame(results)
    # Ordenar por RMSE ascendente (menor es mejor)
    df_sorted = df.dropna(subset=["RMSE"]).sort_values("RMSE")
    df_nan = df[df["RMSE"].isna()]
    df = pd.concat([df_sorted, df_nan]).reset_index(drop=True)
    return df


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
    Función unificada para ejecutar benchmarking según el tipo de problema.

    Args:
        problem_type: 'classification', 'regression', 'timeseries'
        X, y: Para clasificación y regresión
        series: Array/Serie para series de tiempo
        models: dict de modelos (None = defaults)
        test_size: Proporción del test set
        cv_folds: Número de folds K-Fold
        threshold: Umbral de decisión (clasificación)
        balancing_technique: Técnica de balanceo (clasificación)
        scale: Si True, aplica StandardScaler
        seasonal_periods: Períodos estacionales (series de tiempo)
        train_ratio: Ratio de entrenamiento (series de tiempo)
        random_state: Semilla

    Returns:
        dict con 'results' DataFrame y datos de contexto
    """
    problem_type = problem_type.lower()

    if problem_type in ("classification", "regression"):
        stratify = (problem_type == "classification")
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        if scale:
            X_train, X_test, scaler = scale_features(X_train, X_test)
        else:
            scaler = None

        if problem_type == "classification":
            df = benchmark_classification(
                X_train, X_test, y_train, y_test,
                models=models,
                threshold=threshold,
                cv_folds=cv_folds,
                balancing_technique=balancing_technique,
                random_state=random_state,
            )
            return {
                "results": df,
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "scaler": scaler,
                "problem_type": problem_type,
            }

        else:  # regression
            df = benchmark_regression(
                X_train, X_test, y_train, y_test,
                models=models,
                cv_folds=cv_folds,
                random_state=random_state,
            )
            return {
                "results": df,
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "scaler": scaler,
                "problem_type": problem_type,
            }

    elif problem_type == "timeseries":
        import pandas as pd
        if isinstance(series, pd.Series):
            series = series.values.astype(float)

        split_idx = int(len(series) * train_ratio)
        train, test = series[:split_idx], series[split_idx:]

        df = benchmark_timeseries(
            train, test,
            models=models,
            seasonal_periods=seasonal_periods,
        )
        return {
            "results": df,
            "train": train,
            "test": test,
            "problem_type": problem_type,
        }

    else:
        raise ValueError(f"Tipo de problema no válido: '{problem_type}'. "
                         f"Opciones: 'classification', 'regression', 'timeseries'")


def rank_models(df, metric=None, ascending=None):
    """
    Ordena el DataFrame de resultados por la métrica especificada.

    Args:
        df: DataFrame de resultados del benchmarking
        metric: Columna a ordenar (None = detecta automáticamente)
        ascending: True/False (None = auto según métrica)

    Returns:
        DataFrame ordenado
    """
    if metric is None:
        if "AUC-ROC" in df.columns:
            metric = "AUC-ROC"
            ascending = False
        elif "R²" in df.columns:
            metric = "R²"
            ascending = False
        elif "RMSE" in df.columns:
            metric = "RMSE"
            ascending = True
        else:
            return df

    if ascending is None:
        ascending = metric in ("RMSE", "MSE", "MAE", "MAPE (%)")

    return df.sort_values(metric, ascending=ascending).reset_index(drop=True)
