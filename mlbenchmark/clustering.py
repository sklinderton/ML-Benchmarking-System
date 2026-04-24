"""
clustering.py  ·  K-Means Clustering con análisis Elbow y Silhouette
BCD-6210 Minería de Datos - Universidad LEAD
Artículo Científico: Churn Prediction Benchmarking · I Cuatrimestre 2026

Clase KMeansClusterer:
  - fit(X, k): ajusta K-Means para un valor de k dado
  - elbow_analysis(X, k_range): calcula WCSS para rango de k (método del codo)
  - silhouette_analysis(X, k_range): calcula Silhouette Score para rango de k
  - get_cluster_profile(df, label_col): tabla de perfil por cluster

Función de utilidad:
  - cluster_churn_profile(df, cluster_col, churn_col): tasa de churn por cluster
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class KMeansClusterer:
    """
    Envuelve KMeans de sklearn con análisis Elbow + Silhouette y perfil de clusters.

    Workflow:
        clusterer = KMeansClusterer(random_state=42)
        wcss, sil = clusterer.run_analysis(X_scaled, k_range=range(2, 11))
        clusterer.fit(X_scaled, k=3)
        labels = clusterer.labels_
        profile = clusterer.get_cluster_profile(df_original)
    """

    def __init__(self, random_state: int = 42, n_init: int = 10, max_iter: int = 300):
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self._model: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.k_: Optional[int] = None
        self.inertia_: Optional[float] = None

    def fit(self, X: np.ndarray, k: int) -> "KMeansClusterer":
        """
        Ajusta K-Means con k clusters.

        Args:
            X: Array 2D de features (ya escaladas).
            k: Número de clusters.

        Returns:
            self
        """
        self._model = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        self.labels_ = self._model.fit_predict(X)
        self.inertia_ = self._model.inertia_
        self.k_ = k
        return self

    def elbow_analysis(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11),
    ) -> pd.DataFrame:
        """
        Calcula WCSS (inercia) para cada k en k_range (método del codo).

        Args:
            X:       Array 2D de features escaladas.
            k_range: Rango de valores de k a evaluar.

        Returns:
            DataFrame con columnas: k, wcss
        """
        results = []
        for k in k_range:
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            km.fit(X)
            results.append({"k": k, "wcss": km.inertia_})
        return pd.DataFrame(results)

    def silhouette_analysis(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11),
        sample_size: Optional[int] = 5000,
    ) -> pd.DataFrame:
        """
        Calcula Silhouette Score para cada k en k_range.

        Args:
            X:           Array 2D de features escaladas.
            k_range:     Rango de valores de k a evaluar.
            sample_size: Sub-muestra para acelerar el cálculo (None = todos).

        Returns:
            DataFrame con columnas: k, silhouette
        """
        rng = np.random.RandomState(self.random_state)
        if sample_size and len(X) > sample_size:
            idx = rng.choice(len(X), size=sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        results = []
        for k in k_range:
            if k >= len(X_sample):
                continue
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = km.fit_predict(X_sample)
            score = silhouette_score(X_sample, labels, random_state=self.random_state)
            results.append({"k": k, "silhouette": round(score, 4)})
        return pd.DataFrame(results)

    def run_analysis(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta elbow_analysis y silhouette_analysis en un solo paso.

        Returns:
            (df_elbow, df_silhouette)
        """
        return self.elbow_analysis(X, k_range), self.silhouette_analysis(X, k_range)

    def get_cluster_profile(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Genera tabla de medias por cluster sobre las columnas numéricas.

        Args:
            df:            DataFrame original (sin escalar, con las mismas filas que en fit).
            numeric_cols:  Columnas a incluir (None = todas las numéricas).

        Returns:
            DataFrame con índice = cluster y columnas = medias.

        Raises:
            NotFittedError: Si fit() no ha sido llamado.
        """
        if self.labels_ is None:
            raise NotFittedError("Llama fit() primero.")

        df_p = df.copy()
        df_p["_cluster"] = self.labels_

        if numeric_cols is None:
            numeric_cols = df_p.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "_cluster"]

        profile = df_p.groupby("_cluster")[numeric_cols].mean().round(4)
        profile.index.name = "Cluster"
        return profile

    def pca_2d(self, X: np.ndarray) -> pd.DataFrame:
        """
        Reduce X a 2 componentes PCA para visualización.

        Returns:
            DataFrame con columnas: PC1, PC2, Cluster
        """
        if self.labels_ is None:
            raise NotFittedError("Llama fit() primero.")

        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(X)
        return pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": self.labels_.astype(str),
        })


def cluster_churn_profile(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    churn_col: str = "Churn",
) -> pd.DataFrame:
    """
    Calcula la tasa de churn y conteo por cluster.

    Args:
        df:          DataFrame con columna de cluster y columna de churn (0/1 o Yes/No).
        cluster_col: Nombre de la columna de clusters.
        churn_col:   Nombre de la columna de churn.

    Returns:
        DataFrame con columnas: Cluster, N, Churn_N, Churn_Rate(%)
    """
    df_p = df.copy()
    # Normalizar churn a 0/1
    if df_p[churn_col].dtype == object:
        df_p[churn_col] = df_p[churn_col].map(
            lambda v: 1 if str(v).strip().lower() in ("yes", "1", "true") else 0
        )

    grp = df_p.groupby(cluster_col).agg(
        N=(churn_col, "count"),
        Churn_N=(churn_col, "sum"),
    ).reset_index()
    grp["Churn_Rate(%)"] = (grp["Churn_N"] / grp["N"] * 100).round(2)
    return grp


def preprocess_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesamiento estándar del dataset IBM Telco Customer Churn.

    Pasos:
      1. Elimina customerID
      2. Convierte TotalCharges a numérico (blancos → NaN → mediana)
      3. Mapea Yes/No → 1/0 en columnas binarias
      4. Aplica pd.get_dummies(drop_first=True) en columnas multicategoría
      5. Mapea Churn → 0/1 (si no fue binarizado por OHE)

    Args:
        df: DataFrame crudo del CSV Telco.

    Returns:
        df_processed con columna 'Churn' como 0/1 al final.
    """
    df = df.copy()

    # 1. Eliminar identificador
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # 2. TotalCharges: coerción numérica + imputar mediana
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median_tc = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

    # 3. Extraer Churn antes del OHE para preservar el target
    churn_series = None
    if "Churn" in df.columns:
        churn_raw = df.pop("Churn")
        churn_series = churn_raw.map(
            lambda v: 1 if str(v).strip().lower() in ("yes", "1") else 0
        )

    # 4. Binarizar Yes/No directas
    binary_cols = [
        c for c in df.columns
        if df[c].dtype == object and set(df[c].dropna().unique()) <= {"Yes", "No"}
    ]
    for c in binary_cols:
        df[c] = df[c].map({"Yes": 1, "No": 0})

    # 5. SeniorCitizen ya es 0/1; gender → OHE junto con multicategoría
    ohe_cols = df.select_dtypes(include="object").columns.tolist()
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # 6. Recolocar Churn al final
    if churn_series is not None:
        df["Churn"] = churn_series.values

    return df
