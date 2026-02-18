"""
metrics.py - Cálculo de métricas de evaluación para clasificación, regresión y series de tiempo
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)


def classification_metrics(y_true, y_pred, y_prob=None, threshold=0.5):
    """
    Calcula métricas completas de clasificación.

    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones del modelo (clase)
        y_prob: Probabilidades predichas (para AUC)
        threshold: Umbral de decisión ya aplicado al calcular y_pred

    Returns:
        dict con todas las métricas
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            metrics["auc_roc"] = 0.5
    else:
        metrics["auc_roc"] = None

    return metrics


def regression_metrics(y_true, y_pred):
    """
    Calcula métricas de regresión.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        dict con MSE, RMSE, MAE, R²
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def timeseries_metrics(y_true, y_pred):
    """
    Calcula métricas para series de tiempo incluyendo MAPE.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        dict con MSE, RMSE, MAE, MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # MAPE: evitar división por cero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    return {
        "mse": round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "mape": round(mape, 2) if not np.isnan(mape) else None,
    }


def roc_curve_data(y_true, y_prob):
    """
    Calcula puntos de la curva ROC.

    Returns:
        fpr, tpr, thresholds arrays
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist()


def precision_recall_curve_data(y_true, y_prob):
    """
    Calcula curva Precision-Recall.

    Returns:
        precision, recall, thresholds arrays
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision.tolist(), recall.tolist(), thresholds.tolist()
