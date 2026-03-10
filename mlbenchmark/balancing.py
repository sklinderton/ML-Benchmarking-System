"""
balancing.py - Técnicas para manejo de clases desbalanceadas
BCD-7213 Minería de Datos Avanzada - Universidad LEAD

Cambios respecto a la versión anterior:
  - apply_smote(): ajusta k_neighbors automáticamente si la clase
    minoritaria tiene menos de k_neighbors+1 muestras (evita ValueError).
  - apply_combined(): también usa el SMOTE seguro.
"""

import numpy as np
from collections import Counter


def check_imbalance(y):
    """
    Analiza el nivel de desbalanceo en el target.

    Returns:
        dict con clases, conteos, ratio minoritaria/mayoritaria y severidad.
    """
    counter  = Counter(y)
    classes  = sorted(counter.keys())
    counts   = [counter[c] for c in classes]
    minority = min(counts)
    majority = max(counts)
    ratio    = minority / majority

    return {
        "classes":      classes,
        "counts":       counts,
        "total":        len(y),
        "ratio":        round(ratio, 4),
        "is_imbalanced": ratio < 0.3,
        "severity":     "Alto" if ratio < 0.1 else ("Moderado" if ratio < 0.3 else "Bajo"),
    }


def apply_smote(X, y, sampling_strategy="auto", random_state=42, k_neighbors=5):
    """
    Aplica SMOTE para sobre-muestreo de la clase minoritaria.

    Ajusta k_neighbors automáticamente si la clase minoritaria tiene
    menos muestras que k_neighbors+1 (evita ValueError de sklearn).

    Args:
        X: Features de entrenamiento
        y: Target de entrenamiento
        sampling_strategy: 'auto' o float (ratio deseado)
        random_state: Semilla
        k_neighbors: Vecinos para interpolación (se reduce si es necesario)

    Returns:
        X_resampled, y_resampled
        Si la clase minoritaria tiene solo 1 muestra, retorna X, y sin cambios.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("Instala imbalanced-learn:  pip install imbalanced-learn")

    # Calcular k seguro: SMOTE necesita min_samples >= k_neighbors + 1
    classes, counts = np.unique(y, return_counts=True)
    min_samples = int(counts.min())

    safe_k = min(k_neighbors, min_samples - 1)

    if safe_k < 1:
        # Clase minoritaria con 1 sola muestra: SMOTE es imposible
        return X, y

    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=safe_k,
    )
    return sm.fit_resample(X, y)


def undersample(X, y, sampling_strategy="auto", random_state=42):
    """
    Aplica Random Under-Sampling para reducir la clase mayoritaria.

    Args:
        X, y: Datos de entrenamiento
        sampling_strategy: 'auto' o float
        random_state: Semilla

    Returns:
        X_resampled, y_resampled
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
        raise ImportError("Instala imbalanced-learn:  pip install imbalanced-learn")

    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    return rus.fit_resample(X, y)


def apply_combined(X, y, smote_ratio=0.5, under_ratio=1.0, random_state=42):
    """
    Estrategia híbrida: SMOTE → Under-sampling.

    Usa apply_smote() seguro (con ajuste automático de k_neighbors).

    Args:
        X, y: Datos de entrenamiento
        smote_ratio: Ratio objetivo para SMOTE
        under_ratio: Ratio objetivo para under-sampling
        random_state: Semilla

    Returns:
        X_resampled, y_resampled
    """
    X_s, y_s = apply_smote(X, y,
                             sampling_strategy=smote_ratio,
                             random_state=random_state)
    return undersample(X_s, y_s,
                       sampling_strategy=under_ratio,
                       random_state=random_state)


def apply_balancing(X, y, technique="none", random_state=42):
    """
    Función unificada para aplicar cualquier técnica de balanceo.

    Args:
        X, y: Datos de entrenamiento
        technique: 'none' | 'smote' | 'undersample' | 'combined'
        random_state: Semilla

    Returns:
        X_bal, y_bal
    """
    technique = str(technique).lower()

    if technique == "smote":
        return apply_smote(X, y, random_state=random_state)
    elif technique in ("undersample", "under"):
        return undersample(X, y, random_state=random_state)
    elif technique == "combined":
        return apply_combined(X, y, random_state=random_state)
    else:
        return X, y