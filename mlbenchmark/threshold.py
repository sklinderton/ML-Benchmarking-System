"""
threshold.py - Ajuste y optimización de probabilidad de corte (threshold)
"""

import numpy as np


def apply_threshold(y_prob, threshold=0.5):
    """
    Aplica un umbral de decisión a las probabilidades predichas.

    Args:
        y_prob: Array de probabilidades para la clase positiva
        threshold: Valor de corte (0.0 a 1.0)

    Returns:
        y_pred: Array de clases predichas (0 o 1)
    """
    return (np.array(y_prob) >= threshold).astype(int)


def optimize_threshold(y_true, y_prob, metric="f1", thresholds=None):
    """
    Busca el threshold óptimo maximizando una métrica dada.

    Args:
        y_true: Etiquetas reales
        y_prob: Probabilidades para clase positiva
        metric: 'f1', 'recall', 'precision', 'accuracy'
        thresholds: Lista de umbrales a probar (default: 0.01 a 0.99)

    Returns:
        dict con threshold_optimo y score máximo
    """
    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    metric_fns = {
        "f1": f1_score,
        "recall": recall_score,
        "precision": precision_score,
        "accuracy": accuracy_score,
    }

    if metric not in metric_fns:
        raise ValueError(f"Métrica '{metric}' no válida. Opciones: {list(metric_fns.keys())}")

    fn = metric_fns[metric]
    best_threshold = 0.5
    best_score = 0.0
    scores = []

    for t in thresholds:
        y_pred = apply_threshold(y_prob, t)
        try:
            score = fn(y_true, y_pred, zero_division=0)
        except TypeError:
            score = fn(y_true, y_pred)
        scores.append({"threshold": round(float(t), 2), "score": round(float(score), 4)})
        if score > best_score:
            best_score = score
            best_threshold = t

    return {
        "optimal_threshold": round(float(best_threshold), 2),
        "best_score": round(float(best_score), 4),
        "metric": metric,
        "all_scores": scores,
    }


def threshold_analysis(y_true, y_prob, thresholds=None):
    """
    Analiza el efecto del threshold en todas las métricas principales.

    Returns:
        Lista de dicts con {threshold, accuracy, precision, recall, f1}
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    results = []
    for t in thresholds:
        y_pred = apply_threshold(y_prob, t)
        results.append({
            "threshold": round(float(t), 2),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        })

    return results
