"""
validation.py - Validación cruzada K-Fold y estratificada
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


def kfold_validation(model, X, y, k=5, scoring="r2", random_state=42):
    """
    Validación cruzada K-Fold estándar (para regresión).

    Args:
        model: Modelo de sklearn
        X: Features
        y: Target
        k: Número de folds
        scoring: Métrica a calcular
        random_state: Semilla

    Returns:
        dict con mean, std, scores
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return {
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "scores": [round(float(s), 4) for s in scores],
        "k": k,
        "scoring": scoring,
    }


def stratified_kfold(model, X, y, k=5, scoring="roc_auc", random_state=42):
    """
    Validación cruzada K-Fold Estratificada (para clasificación).
    Mantiene la proporción de clases en cada fold.

    Args:
        model: Modelo de sklearn
        X: Features
        y: Target
        k: Número de folds
        scoring: Métrica ('roc_auc', 'accuracy', 'f1', etc.)
        random_state: Semilla

    Returns:
        dict con mean, std, scores
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return {
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "scores": [round(float(s), 4) for s in scores],
        "k": k,
        "scoring": scoring,
    }


def manual_kfold(model, X, y, k=5, problem_type="classification", random_state=42):
    """
    K-Fold manual con cálculo de múltiples métricas por fold.
    Útil cuando se necesitan varias métricas simultáneamente.

    Args:
        model: Modelo de sklearn
        X, y: Datos
        k: Número de folds
        problem_type: 'classification' o 'regression'
        random_state: Semilla

    Returns:
        Lista de dicts con métricas por fold
    """
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  r2_score, mean_squared_error)

    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)

        if problem_type == "classification":
            y_pred = model.predict(X_val)
            fold_res = {
                "fold": fold,
                "accuracy": round(accuracy_score(y_val, y_pred), 4),
                "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
            }
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
                try:
                    fold_res["auc"] = round(roc_auc_score(y_val, y_prob), 4)
                except Exception:
                    fold_res["auc"] = 0.5
        else:
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            fold_res = {
                "fold": fold,
                "r2": round(r2_score(y_val, y_pred), 4),
                "rmse": round(float(np.sqrt(mse)), 4),
            }

        fold_results.append(fold_res)

    return fold_results
