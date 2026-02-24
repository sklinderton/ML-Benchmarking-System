"""
eda_streamlit_adapter.py
Adaptador que conecta la clase analisisEDA (PaqEda.py) con Streamlit.

El paquete original usa plt.show() y print(). 
Este adaptador:
  - Recibe un DataFrame directamente (sin ruta de archivo)
  - Redirige plt.show() → st.pyplot()
  - Captura los print() → los retorna como strings
  - Expone cada método como función llamable desde la interfaz
"""

import io
import contextlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sin ventana (compatible con Streamlit)
import matplotlib.pyplot as plt
import streamlit as st


# ── Inyectar el DataFrame directamente en analisisEDA ─────────────────────────
# La clase original solo acepta una ruta CSV. Creamos una subclase que
# permite pasar un DataFrame en memoria sin tocar el código original.

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from PaqEda import analisisEDA as _analisisEDA_orig


class analisisEDA_Streamlit(_analisisEDA_orig):
    """
    Subclase de analisisEDA que acepta un pd.DataFrame directamente
    y reemplaza plt.show() por st.pyplot() para compatibilidad con Streamlit.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe: pd.DataFrame ya cargado (de Streamlit o predeterminado)
        """
        # Saltamos el __init__ original (que necesita ruta de archivo)
        # e inyectamos el df directamente usando el setter público
        object.__setattr__(self, '_analisisEDA__df', dataframe.copy())

    # ── helpers internos ────────────────────────────────────────────────────

    @staticmethod
    def _capture_print(func, *args, **kwargs):
        """Captura cualquier print() de una función y lo retorna como string."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = func(*args, **kwargs)
        return buf.getvalue(), result

    @staticmethod
    def _show_fig():
        """Renderiza la figura actual en Streamlit y la cierra."""
        st.pyplot(plt.gcf())
        plt.close("all")

    # ── Métodos adaptados para Streamlit ────────────────────────────────────

    def st_tipoDatos(self):
        df = self.df
        return pd.DataFrame({"Columna": df.columns, "Tipo": df.dtypes.values.astype(str)})

    def st_analisis(self):
        """Retorna estadísticas descriptivas como DataFrame."""
        df = self.df
        num = df.select_dtypes(include="number")
        stats = pd.DataFrame({
            "Media":     num.mean(),
            "Mediana":   num.median(),
            "Std":       num.std(),
            "Mínimo":    num.min(),
            "Máximo":    num.max(),
            "Q1 (25%)":  num.quantile(0.25),
            "Q3 (75%)":  num.quantile(0.75),
        }).round(4)
        return stats

    def st_valores_faltantes(self):
        """Retorna DataFrame con nulos por columna."""
        df = self.df
        nulos = df.isna().sum()
        pct   = (nulos / len(df) * 100).round(2)
        return pd.DataFrame({
            "Columna":       nulos.index,
            "Nulos":         nulos.values,
            "Porcentaje_%":  pct.values,
        }).query("Nulos > 0").reset_index(drop=True)

    def st_valores_unicos(self, col):
        """Retorna DataFrame con conteo de valores únicos para una columna."""
        vc = self.df[col].value_counts().reset_index()
        vc.columns = ["Valor", "Conteo"]
        vc["Porcentaje_%"] = (vc["Conteo"] / len(self.df) * 100).round(2)
        return vc

    def st_eliminarDuplicados(self):
        """Elimina duplicados y retorna mensaje."""
        antes = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        despues = self.df.shape[0]
        return f"✅ Se eliminaron **{antes - despues}** filas duplicadas. Total actual: **{despues}** filas."

    def st_eliminarNulos(self):
        """Elimina filas con nulos y retorna mensaje."""
        nulos_antes  = self.df.isnull().sum().sum()
        filas_antes  = self.df.shape[0]
        self.df      = self.df.dropna()
        filas_despues = self.df.shape[0]
        return (f"✅ Nulos eliminados: **{nulos_antes}** | "
                f"Filas eliminadas: **{filas_antes - filas_despues}** | "
                f"Filas restantes: **{filas_despues}**")

    def st_graficoBoxplot(self):
        """Boxplots de variables numéricas → st.pyplot()"""
        self.graficoBoxplot()
        self._show_fig()

    def st_histogramas(self):
        """Histogramas básicos → st.pyplot()"""
        self.histogramas()
        self._show_fig()

    def st_distribucionVariables(self):
        """Histogramas con KDE (distribucionVariables) → st.pyplot()"""
        self.distribucionVariables()
        self._show_fig()

    def st_histogramaClase(self, columna_objetivo):
        """Distribución de la clase objetivo → st.pyplot()"""
        self.histogramaClase(columna_objetivo)
        self._show_fig()

    def st_datosDensidad(self):
        """Gráficos de densidad KDE → st.pyplot()"""
        self.datosDensidad()
        self._show_fig()

    def st_graficoCorrelacion(self):
        """Heatmap de correlación → st.pyplot()"""
        self.graficoCorrelacion()
        self._show_fig()

    def st_graficosDispersion(self, max_cols=6):
        """
        Pairplot limitado a max_cols columnas numéricas
        (pairplot completo puede ser muy lento con muchas columnas).
        """
        import seaborn as sns
        num_cols = self.df.select_dtypes(include="number").columns.tolist()[:max_cols]
        if len(num_cols) < 2:
            st.warning("Se necesitan al menos 2 columnas numéricas.")
            return
        fig = plt.figure(figsize=(10, 8))
        pair = sns.pairplot(self.df[num_cols])
        st.pyplot(pair.figure)
        plt.close("all")

    def st_correlaciones(self):
        """Retorna la matriz de correlación como DataFrame."""
        return self.df.corr(numeric_only=True).round(4)
