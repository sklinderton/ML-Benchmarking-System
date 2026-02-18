"""
mlbenchmark - Sistema de Benchmarking de Modelos de Machine Learning
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
"""

from .benchmarking import run_benchmark, rank_models
from .preprocessing import split_data, scale_features
from .balancing import apply_smote, undersample, check_imbalance
from .validation import kfold_validation, stratified_kfold
from .metrics import classification_metrics, regression_metrics, timeseries_metrics
from .threshold import apply_threshold, optimize_threshold

__version__ = "1.0.0"
__all__ = [
    "run_benchmark", "rank_models",
    "split_data", "scale_features",
    "apply_smote", "undersample", "check_imbalance",
    "kfold_validation", "stratified_kfold",
    "classification_metrics", "regression_metrics", "timeseries_metrics",
    "apply_threshold", "optimize_threshold",
]
