"""
eda.py - Paquete de Análisis Exploratorio de Datos (EDA)
Arquitectura CRISP-DM + OOP + Principios SOLID
BCD-7213 Minería de Datos Avanzada - Universidad LEAD

Adaptaciones para Streamlit:
  - Métodos gráficos retornan fig (compatible con st.pyplot)
  - Métodos de texto retornan dict/DataFrame (compatible con st.dataframe)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


class analisisEDA:
    """
    Clase principal de EDA siguiendo fases CRISP-DM.
    - Data Understanding: perfilado, estadísticas, nulos, duplicados
    - Data Preparation: limpieza, dummies, renombrado
    - Visualización: boxplots, histogramas, correlaciones, dispersión
    """

    def __init__(self, df: pd.DataFrame):
        self.__df = df.copy()

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, nuevo_df: pd.DataFrame):
        self.__df = nuevo_df.copy()

    # ── Data Understanding ───────────────────────────────────────

    def tipoDatos(self) -> pd.DataFrame:
        """Retorna tabla con nombre y tipo de cada columna."""
        return pd.DataFrame({
            "Columna": self.__df.dtypes.index,
            "Tipo": self.__df.dtypes.values.astype(str),
            "No Nulos": self.__df.count().values,
            "Nulos": self.__df.isna().sum().values,
        }).reset_index(drop=True)

    def analisis(self) -> dict:
        """Estadísticas descriptivas completas. Retorna dict."""
        num = self.__df.select_dtypes(include="number")
        return {
            "dimensiones": self.__df.shape,
            "head": self.__df.head(10),
            "media": num.mean(),
            "mediana": num.median(),
            "std": num.std(),
            "maximo": num.max(),
            "minimo": num.min(),
            "cuantiles": num.quantile([0, 0.25, 0.5, 0.75, 1]),
        }

    def valores_faltantes(self) -> pd.DataFrame:
        """Retorna tabla de nulos por columna."""
        missing = self.__df.isna().sum()
        pct = (missing / len(self.__df) * 100).round(2)
        df_out = pd.DataFrame({
            "Columna": missing.index,
            "Nulos": missing.values,
            "Porcentaje (%)": pct.values,
        })
        return df_out[df_out["Nulos"] > 0].reset_index(drop=True)

    def valores_unicos(self, columna: str) -> pd.DataFrame:
        """Frecuencia de valores únicos en una columna."""
        vc = self.__df[columna].value_counts().reset_index()
        vc.columns = ["Valor", "Conteo"]
        return vc

    # ── Data Preparation ─────────────────────────────────────────

    def eliminarDuplicados(self) -> int:
        antes = len(self.__df)
        self.__df.drop_duplicates(inplace=True)
        return antes - len(self.__df)

    def eliminarNulos(self) -> dict:
        nulos_antes = int(self.__df.isnull().sum().sum())
        filas_antes = len(self.__df)
        self.__df.dropna(inplace=True)
        return {
            "nulos_eliminados": nulos_antes,
            "filas_eliminadas": filas_antes - len(self.__df),
            "filas_restantes": len(self.__df),
        }

    def eliminarColumnas(self, columnas: list):
        self.__df.drop(columns=columnas, inplace=True, errors="ignore")

    def renombrarColumnas(self, nuevos_nombres: dict):
        self.__df.rename(columns=nuevos_nombres, inplace=True)

    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include=["number"])

    def analisisCompleto(self) -> list:
        cols_cat = self.__df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cols_cat:
            self.__df = pd.get_dummies(self.__df, columns=cols_cat, drop_first=True).astype(int)
        return cols_cat

    # ── Visualizaciones ──────────────────────────────────────────

    def _setup_grid(self, n: int, ncols: int = 3):
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4 * nrows), dpi=110)
        axes = np.array(axes).flatten()
        return fig, axes

    def graficoBoxplot(self) -> plt.Figure:
        cols = self.__df.select_dtypes(include="number").columns
        n = len(cols)
        if n == 0:
            return None
        fig, axes = self._setup_grid(n)
        palette = sns.color_palette("Set3", n)
        for i, col in enumerate(cols):
            sns.boxplot(y=self.__df[col], ax=axes[i], color=palette[i])
            axes[i].set_title(f"Boxplot: {col}", fontsize=9)
            axes[i].grid(True, linestyle="--", alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def histogramas(self) -> plt.Figure:
        cols = self.__df.select_dtypes(include="number").columns
        n = len(cols)
        if n == 0:
            return None
        fig, axes = self._setup_grid(n)
        palette = sns.color_palette("Set2", n)
        for i, col in enumerate(cols):
            axes[i].hist(self.__df[col], bins=30, color=palette[i],
                         edgecolor="black", alpha=0.7)
            axes[i].set_title(f"Histograma: {col}", fontsize=9)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle="--", alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def distribucionVariables(self) -> plt.Figure:
        cols = self.__df.select_dtypes(include="number").columns
        n = len(cols)
        if n == 0:
            return None
        fig, axes = self._setup_grid(n)
        palette = sns.color_palette("coolwarm", n)
        for i, col in enumerate(cols):
            sns.histplot(self.__df[col], kde=True, ax=axes[i],
                         color=palette[i], bins=30)
            axes[i].set_title(f"Distribución: {col}", fontsize=9)
            axes[i].grid(True, linestyle="--", alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def datosDensidad(self) -> plt.Figure:
        cols = self.__df.select_dtypes(include="number").columns
        n = len(cols)
        if n == 0:
            return None
        fig, axes = self._setup_grid(n)
        palette = sns.color_palette("husl", n)
        for i, col in enumerate(cols):
            sns.kdeplot(data=self.__df, x=col, fill=True,
                        ax=axes[i], color=palette[i], linewidth=2)
            axes[i].set_title(f"Densidad: {col}", fontsize=9)
            axes[i].grid(True, linestyle="--", alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def histogramaClase(self, columna_objetivo: str) -> plt.Figure:
        if columna_objetivo not in self.__df.columns:
            return None
        fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
        palette = sns.color_palette("pastel")
        self.__df[columna_objetivo].value_counts().plot(
            kind="bar", color=palette, ax=ax
        )
        ax.set_title(f"Distribución de la Clase: {columna_objetivo}")
        ax.set_xlabel(columna_objetivo)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig

    def graficoCorrelacion(self) -> plt.Figure:
        corr = self.__df.corr(numeric_only=True)
        if corr.empty:
            return None
        fig, ax = plt.subplots(figsize=(12, 8), dpi=110)
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(
            corr, vmin=-1, vmax=1, cmap=cmap, annot=True,
            fmt=".2f", linewidths=0.5, linecolor="white",
            square=True, cbar_kws={"shrink": 0.8, "label": "Correlación"},
            annot_kws={"size": 8, "color": "black"}, ax=ax
        )
        ax.set_title("Mapa de Calor de Correlaciones", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def correlaciones(self) -> pd.DataFrame:
        return self.__df.corr(numeric_only=True).round(4)

    def graficosDispersion(self) -> plt.Figure:
        cols = self.__df.select_dtypes(include="number").columns[:8]
        if len(cols) < 2:
            return None
        sample = self.__df[cols].sample(min(500, len(self.__df)), random_state=42)
        pair = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.4})
        pair.fig.suptitle("Gráficos de Dispersión por Pares", y=1.02, fontsize=12)
        return pair.fig
