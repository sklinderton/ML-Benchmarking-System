"""
mlbenchmark - Sistema de Benchmarking de Modelos de Machine Learning
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
"""

from .benchmarking import run_benchmark, rank_models
from .preprocessing import split_data, scale_features, impute_missing, encode_categorical_ohe
from .clustering import KMeansClusterer, cluster_churn_profile, preprocess_telco
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
    "impute_missing", "encode_categorical_ohe",
    "KMeansClusterer", "cluster_churn_profile", "preprocess_telco",
]
