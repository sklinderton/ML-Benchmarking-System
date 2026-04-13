"""
eda.py - Paquete de Análisis Exploratorio de Datos (EDA)
Arquitectura CRISP-DM + OOP + Principios SOLID
BCD-7213 Minería de Datos Avanzada - Universidad LEAD

MEJORA: analisisEDA detecta automáticamente la columna target candidata
  mediante el método sugerir_target(), que busca columnas con pocas
  categorías únicas, desbalance detectable y alta correlación con el resto.
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
    MEJORA: sugerir_target() detecta automáticamente columna objetivo candidata
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

    # ── MEJORA: detección automática de target ───────────────────

    def sugerir_target(self, max_categorias: int = 20) -> dict:
        """
        MEJORA: detecta automáticamente la columna más probable como target.

        Criterios de puntuación (mayor score → mejor candidato):
          +3 si la columna se llama 'target', 'label', 'clase', 'y', 'output'
          +2 si es la última columna del DataFrame
          +2 si tiene pocas categorías únicas (≤ max_categorias) → clasificación
          +1 si hay desbalance detectable (ratio min/max < 0.4)
          +1 si es columna numérica binaria (0/1)
          +1 si su nombre contiene palabras clave de target
          -1 si tiene demasiados valores únicos (probable ID)
          -1 si contiene más del 30% de nulos

        Args:
            max_categorias: máximo de valores únicos para considerar clasificación

        Returns:
            dict con:
              'columna_sugerida': str  — nombre de la columna recomendada
              'tipo_problema':    str  — 'classification' o 'regression'
              'razon':            str  — explicación legible
              'scores':           dict — puntuación de cada columna
              'detalle':          pd.DataFrame — tabla completa de análisis
        """
        df = self.__df
        n_filas = len(df)
        scores = {}
        detalles = []

        # MEJORA: palabras clave que sugieren columna target
        KEYWORDS_TARGET = {
            "target", "label", "clase", "class", "y", "output",
            "resultado", "etiqueta", "categoria", "category",
            "fraude", "fraud", "churn", "default", "survived",
            "outcome", "diagnosis", "response", "precio", "price",
            "salary", "income", "ingreso", "sales", "ventas",
        }

        for col in df.columns:
            score = 0
            n_unicos = df[col].nunique()
            pct_nulos = df[col].isna().mean()
            col_lower = col.lower().strip()

            # MEJORA: criterio nombre exacto
            if col_lower in KEYWORDS_TARGET:
                score += 3

            # MEJORA: criterio última columna (convención común)
            if col == df.columns[-1]:
                score += 2

            # MEJORA: criterio nombre parcial
            if any(kw in col_lower for kw in KEYWORDS_TARGET):
                score += 1

            # MEJORA: pocas categorías → candidato a clasificación
            if 2 <= n_unicos <= max_categorias:
                score += 2

            # MEJORA: columna numérica binaria (0/1 o True/False)
            if df[col].dropna().isin([0, 1, True, False]).all() and n_unicos == 2:
                score += 1

            # MEJORA: desbalance detectable → clasificación real
            if 2 <= n_unicos <= max_categorias:
                vc = df[col].value_counts()
                if len(vc) >= 2:
                    ratio = vc.iloc[-1] / vc.iloc[0]
                    if ratio < 0.4:
                        score += 1

            # MEJORA: penalizar columnas con muchos valores únicos (probable ID)
            if n_unicos > n_filas * 0.9:
                score -= 1

            # MEJORA: penalizar columnas con muchos nulos
            if pct_nulos > 0.3:
                score -= 1

            scores[col] = score
            detalles.append({
                "Columna":    col,
                "Score":      score,
                "N_Únicos":   n_unicos,
                "Tipo_dato":  str(df[col].dtype),
                "% Nulos":    round(pct_nulos * 100, 1),
                "Última_col": col == df.columns[-1],
            })

        # MEJORA: elegir columna con mayor score
        mejor_col = max(scores, key=scores.get)
        mejor_score = scores[mejor_col]
        n_unicos_mejor = df[mejor_col].nunique()

        # MEJORA: determinar tipo de problema
        if n_unicos_mejor <= max_categorias:
            tipo = "classification"
        else:
            tipo = "regression"

        # MEJORA: construir razón legible
        razon_parts = []
        col_lower = mejor_col.lower()
        if any(kw in col_lower for kw in KEYWORDS_TARGET):
            razon_parts.append("nombre sugiere variable objetivo")
        if mejor_col == df.columns[-1]:
            razon_parts.append("es la última columna")
        if 2 <= n_unicos_mejor <= max_categorias:
            razon_parts.append(
                f"{n_unicos_mejor} valores únicos → apta para clasificación")
        else:
            razon_parts.append(
                f"{n_unicos_mejor} valores únicos → apta para regresión")
        razon = "; ".join(razon_parts) if razon_parts else "mayor score compuesto"

        detalle_df = pd.DataFrame(detalles).sort_values("Score", ascending=False)\
                       .reset_index(drop=True)

        return {
            "columna_sugerida": mejor_col,
            "tipo_problema":    tipo,
            "razon":            razon,
            "score":            mejor_score,
            "scores":           scores,
            "detalle":          detalle_df,
        }

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