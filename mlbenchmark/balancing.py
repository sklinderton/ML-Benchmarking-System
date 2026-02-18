"""
balancing.py - Técnicas para manejo de clases desbalanceadas
"""

import numpy as np
from collections import Counter


def check_imbalance(y):
    """
    Analiza el nivel de desbalanceo en el target.

    Returns:
        dict con conteos, ratio minoritaria/mayoritaria e info
    """
    counter = Counter(y)
    classes = sorted(counter.keys())
    counts = [counter[c] for c in classes]
    minority_count = min(counts)
    majority_count = max(counts)
    ratio = minority_count / majority_count

    return {
        "classes": classes,
        "counts": counts,
        "total": len(y),
        "ratio": round(ratio, 4),
        "is_imbalanced": ratio < 0.3,
        "severity": "Alto" if ratio < 0.1 else ("Moderado" if ratio < 0.3 else "Bajo"),
    }


def apply_smote(X, y, sampling_strategy="auto", random_state=42, k_neighbors=5):
    """
    Aplica SMOTE para sobre-muestreo de clase minoritaria.

    Args:
        X: Features de entrenamiento
        y: Target de entrenamiento
        sampling_strategy: 'auto' o float (ratio deseado)
        random_state: Semilla
        k_neighbors: Vecinos para interpolación

    Returns:
        X_resampled, y_resampled
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors
        )
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except ImportError:
        raise ImportError("Instala imbalanced-learn: pip install imbalanced-learn")


def undersample(X, y, sampling_strategy="auto", random_state=42):
    """
    Aplica Random Under-Sampling para reducir clase mayoritaria.

    Args:
        X, y: Datos de entrenamiento
        sampling_strategy: 'auto' o float
        random_state: Semilla

    Returns:
        X_resampled, y_resampled
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_res, y_res = rus.fit_resample(X, y)
        return X_res, y_res
    except ImportError:
        raise ImportError("Instala imbalanced-learn: pip install imbalanced-learn")


def apply_combined(X, y, smote_ratio=0.5, under_ratio=1.0, random_state=42):
    """
    Estrategia híbrida: SMOTE seguido de Under-sampling.

    Args:
        X, y: Datos de entrenamiento
        smote_ratio: Ratio objetivo para SMOTE
        under_ratio: Ratio objetivo para under-sampling
        random_state: Semilla

    Returns:
        X_resampled, y_resampled
    """
    X_smote, y_smote = apply_smote(X, y,
                                   sampling_strategy=smote_ratio,
                                   random_state=random_state)
    X_res, y_res = undersample(X_smote, y_smote,
                               sampling_strategy=under_ratio,
                               random_state=random_state)
    return X_res, y_res


def apply_balancing(X, y, technique="none", random_state=42):
    """
    Función unificada para aplicar cualquier técnica de balanceo.

    Args:
        X, y: Datos de entrenamiento
        technique: 'none', 'smote', 'undersample', 'combined'
        random_state: Semilla

    Returns:
        X_bal, y_bal
    """
    technique = technique.lower()

    if technique == "smote":
        return apply_smote(X, y, random_state=random_state)
    elif technique in ("undersample", "under"):
        return undersample(X, y, random_state=random_state)
    elif technique == "combined":
        return apply_combined(X, y, random_state=random_state)
    else:
        return X, y
