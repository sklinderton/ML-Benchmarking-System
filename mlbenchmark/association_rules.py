"""
association_rules.py  ·  Reglas de Asociación (Apriori)
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
Caso de Estudio 2  ·  I Cuatrimestre 2026

Escenario: Análisis de canasta de compras para la tienda de aventura outdoor.
           Detecta qué categorías/productos se compran juntos frecuentemente
           para generar recomendaciones de cross-selling.

Clase AssociationRulesMiner:
  · Carga de transacciones (listas o DataFrame)
  · Codificación binaria con TransactionEncoder
  · Identificación de itemsets frecuentes (Apriori)
  · Generación de reglas (confianza, lift, soporte)
  · Motor de recomendaciones: dado un ítem → sugiere N más probables

Funciones de utilidad:
    generate_synthetic_transactions(productos_df) → list[list[str]]
    load_groceries_dataset()                      → pd.DataFrame
"""

from __future__ import annotations

import contextlib
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError


# ──────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────

GROCERIES_URL = (
    "https://raw.githubusercontent.com/JoaquinAmatRodrigo/"
    "Estadistica-con-R/master/datos/datos_groceries.csv"
)

_METRIC_LITERAL = Literal["confidence", "lift", "support", "leverage", "conviction"]
_SIDE_LITERAL   = Literal["antecedents", "consequents", "any"]


# ──────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ──────────────────────────────────────────────────────────────

def _require_mlxtend() -> None:
    """
    Verifica que mlxtend esté instalado.

    Raises:
        ImportError: Si mlxtend no está disponible.
    """
    try:
        import importlib
        importlib.import_module("mlxtend")
    except ImportError:
        raise ImportError(
            "mlxtend es requerido. Instálalo con:  pip install mlxtend"
        )


def _frozenset_to_str(fs) -> str:
    """Convierte un frozenset a string legible ordenado."""
    return ", ".join(sorted(str(x) for x in fs))


def _normalize_frozensets(series: pd.Series) -> pd.Series:
    """
    Normaliza frozensets para garantizar que los elementos sean str Python puro.

    Necesario por un bug de mlxtend 0.24 que produce np.str_ en lugar de str,
    lo que rompe association_rules() internamente.
    """
    return series.apply(lambda fs: frozenset(str(x) for x in fs))


# ──────────────────────────────────────────────────────────────
# CARGA DE DATASETS
# ──────────────────────────────────────────────────────────────

def load_groceries_dataset() -> pd.DataFrame:
    """
    Carga el dataset público de comestibles (groceries).

    Returns:
        DataFrame con columnas ['id_compra', 'item'].
        Usa datos sintéticos de aventura outdoor si no hay conexión.
    """
    with contextlib.suppress(Exception):
        df = pd.read_csv(GROCERIES_URL)
        if {"id_compra", "item"}.issubset(df.columns):
            return df
    return _synthetic_groceries()


def _synthetic_groceries() -> pd.DataFrame:
    """
    Genera transacciones sintéticas de productos outdoor con co-ocurrencias realistas.

    Returns:
        DataFrame con columnas ['id_compra', 'item'].
    """
    rng = np.random.RandomState(42)

    # Catálogo base — "Pilas" añadida para coherencia con las co-ocurrencias
    categorias = [
        "Sombrero", "Mochila", "Botas", "Linterna", "Cantimplora",
        "Guantes", "Chaleco", "Sleeping Bag", "Bastones", "Navaja",
        "Brújula", "Cuerda", "Calcetines", "Carpa", "Gorra", "Pilas",
    ]

    cooccurrence: dict[str, list[str]] = {
        "Mochila":      ["Cantimplora", "Linterna", "Botas"],
        "Botas":        ["Calcetines", "Bastones"],
        "Carpa":        ["Sleeping Bag", "Linterna", "Mochila"],
        "Sleeping Bag": ["Carpa", "Linterna"],
        "Linterna":     ["Pilas", "Mochila"],
        "Guantes":      ["Chaleco", "Sombrero"],
    }

    rows = []
    for compra_id in range(1, 2001):
        n_items = rng.randint(2, 7)  # mínimo 2 para generar reglas
        items = list(
            rng.choice(categorias, size=min(n_items, len(categorias)), replace=False)
        )
        for item in items[:]:
            if item in cooccurrence and rng.random() < 0.4:
                co = rng.choice(cooccurrence[item])
                if co not in items:
                    items.append(co)
        for item in items:
            rows.append({"id_compra": compra_id, "item": item})

    return pd.DataFrame(rows)


def generate_synthetic_transactions(
    productos_df: pd.DataFrame,
    n_transacciones: int = 1000,
    max_items_por_compra: int = 6,
    seed: int = 42,
) -> list[list[str]]:
    """
    Genera transacciones sintéticas a partir de un DataFrame de productos.

    Simula un escenario de market basket analysis usando las categorías
    del dataset de Web Mining.

    Args:
        productos_df: DataFrame con columna 'Categoria' o 'Nombre'.
        n_transacciones: Número de transacciones a generar.
        max_items_por_compra: Máximo de productos por transacción.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Lista de listas de strings; cada sublista es una transacción.
    """
    rng = np.random.RandomState(seed)

    if "Categoria" in productos_df.columns:
        items_pool = productos_df["Categoria"].dropna().unique().tolist()
    elif "Nombre" in productos_df.columns:
        items_pool = productos_df["Nombre"].dropna().head(50).tolist()
    else:
        items_pool = productos_df.iloc[:, 0].dropna().head(50).tolist()

    if len(items_pool) < 2:
        items_pool = ["Item_A", "Item_B", "Item_C", "Item_D", "Item_E"]

    min_items = min(2, len(items_pool))
    max_items = min(max_items_por_compra + 1, len(items_pool) + 1)

    transacciones: list[list[str]] = []
    for _ in range(n_transacciones):
        n_items = rng.randint(min_items, max_items)
        compra = list(rng.choice(items_pool, size=n_items, replace=False))
        transacciones.append(compra)

    return transacciones


# ──────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ──────────────────────────────────────────────────────────────

class AssociationRulesMiner:
    """
    Motor de Reglas de Asociación basado en el algoritmo Apriori (mlxtend).

    Workflow típico:
        miner = AssociationRulesMiner()
        miner.fit(transacciones)
        df_itemsets = miner.get_frequent_itemsets(min_support=0.02)
        df_rules    = miner.get_rules(min_confidence=0.3, min_lift=1.0)
        recomend    = miner.recommend("Mochila", top_n=5)

    Attributes:
        _transacciones: Lista original de transacciones.
        _encoded_df:    Matriz binaria item × transacción.
        _itemsets:      DataFrame de itemsets frecuentes (post-fit).
        _rules:         DataFrame de reglas generadas (post-get_rules).
        _last_min_support: Soporte mínimo usado en el último get_frequent_itemsets,
                           para detectar cuando es necesario recalcular.
    """

    def __init__(self) -> None:
        _require_mlxtend()
        self._transacciones: Optional[list[list[str]]] = None
        self._encoded_df: Optional[pd.DataFrame] = None
        self._itemsets: Optional[pd.DataFrame] = None
        self._rules: Optional[pd.DataFrame] = None
        self._fitted: bool = False
        self._last_min_support: float = 0.0

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "not fitted"
        n_rules = len(self._rules) if self._rules is not None else 0
        return (
            f"AssociationRulesMiner({state}, "
            f"transactions={len(self._transacciones) if self._transacciones else 0}, "
            f"rules={n_rules})"
        )

    # ── Validación de estado ───────────────────────────────────

    def _check_fitted(self) -> None:
        """
        Raises:
            NotFittedError: Si fit() no ha sido llamado aún.
        """
        if not self._fitted:
            raise NotFittedError(
                "AssociationRulesMiner no ha sido entrenado. Llama .fit() primero."
            )

    def _check_rules(self) -> None:
        """
        Raises:
            RuntimeError: Si get_rules() no ha sido llamado aún.
        """
        if self._rules is None:
            raise RuntimeError(
                "No hay reglas calculadas. Llama .get_rules() primero."
            )

    # ── Entrenamiento ─────────────────────────────────────────

    def fit(self, transacciones: list[list[str]]) -> "AssociationRulesMiner":
        """
        Codifica las transacciones en matriz binaria y prepara el encoder.

        Args:
            transacciones: Lista de listas de strings.
                           Ejemplo: [["Mochila", "Botas"], ["Linterna", "Carpa"]]

        Returns:
            self (para encadenamiento de métodos).

        Raises:
            ValueError: Si la lista de transacciones está vacía.
        """
        if not transacciones:
            raise ValueError("La lista de transacciones está vacía.")

        from mlxtend.preprocessing import TransactionEncoder

        te = TransactionEncoder()
        matrix = te.fit(transacciones).transform(transacciones)

        # Forzar str Python puro en nombres de columnas (bug mlxtend 0.24 con np.str_)
        self._encoded_df = pd.DataFrame(
            matrix, columns=[str(c) for c in te.columns_]
        )
        self._transacciones = transacciones
        self._itemsets = None   # invalidar caché
        self._rules = None
        self._last_min_support = 0.0
        self._fitted = True
        return self

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        id_col: str = "id_compra",
        item_col: str = "item",
    ) -> "AssociationRulesMiner":
        """
        Construye transacciones desde un DataFrame largo (una fila = un ítem).

        Args:
            df:       DataFrame con al menos las columnas id_col e item_col.
            id_col:   Columna que identifica la transacción (compra).
            item_col: Columna con el nombre del ítem.

        Returns:
            self

        Raises:
            ValueError: Si las columnas requeridas no existen en df.
        """
        missing = {id_col, item_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"Columnas faltantes en el DataFrame: {missing}. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        transacciones = df.groupby(id_col)[item_col].apply(list).tolist()
        return self.fit(transacciones)

    # ── Itemsets frecuentes ───────────────────────────────────

    def get_frequent_itemsets(
        self,
        min_support: float = 0.02,
        max_len: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Identifica itemsets frecuentes con soporte ≥ min_support (Apriori).

        Si se llama con un min_support distinto al último usado, recalcula
        automáticamente. También invalida las reglas en caché.

        Args:
            min_support: Fracción mínima de transacciones (0.0–1.0).
            max_len:     Longitud máxima del itemset. None = sin límite.

        Returns:
            DataFrame con columnas: support, itemsets, itemsets_str, n_items.

        Raises:
            NotFittedError: Si fit() no ha sido llamado.
            ValueError: Si min_support está fuera del rango (0, 1].
        """
        self._check_fitted()

        if not (0.0 < min_support <= 1.0):
            raise ValueError(
                f"min_support debe estar en (0, 1]. Recibido: {min_support}"
            )

        from mlxtend.frequent_patterns import apriori

        kwargs: dict = {"min_support": min_support, "use_colnames": True}
        if max_len is not None:
            kwargs["max_len"] = max_len

        itemsets = apriori(self._encoded_df, **kwargs)

        if itemsets.empty:
            self._itemsets = itemsets
            self._rules = None  # invalidar reglas en caché
            self._last_min_support = min_support
            return itemsets

        # Normalizar frozensets (bug mlxtend 0.24)
        itemsets["itemsets"] = _normalize_frozensets(itemsets["itemsets"])
        itemsets["n_items"] = itemsets["itemsets"].apply(len)
        itemsets["itemsets_str"] = itemsets["itemsets"].apply(_frozenset_to_str)

        self._itemsets = (
            itemsets
            .sort_values("support", ascending=False)
            .reset_index(drop=True)
        )
        self._rules = None  # invalidar reglas en caché cuando cambian los itemsets
        self._last_min_support = min_support
        return self._itemsets

    # ── Reglas de asociación ──────────────────────────────────

    def get_rules(
        self,
        min_confidence: float = 0.3,
        min_lift: float = 1.0,
        min_support: float = 0.01,
        metric: _METRIC_LITERAL = "confidence",
    ) -> pd.DataFrame:
        """
        Genera reglas de asociación a partir de los itemsets frecuentes.

        Si los itemsets no han sido calculados, o si min_support es menor
        al soporte mínimo usado anteriormente, los recalcula primero.

        Args:
            min_confidence: Confianza mínima para las reglas (0–1).
            min_lift:       Lift mínimo; > 1 indica asociación positiva.
            min_support:    Soporte mínimo para itemsets. Recalcula si es
                            menor al soporte previamente usado.
            metric:         Métrica base para mlxtend ('confidence' | 'lift' |
                            'support' | 'leverage' | 'conviction').

        Returns:
            DataFrame con columnas:
                antecedents, consequents, support, confidence, lift,
                leverage, conviction, antecedents_str, consequents_str.
            DataFrame vacío si no se generan reglas con los parámetros dados.

        Raises:
            NotFittedError: Si fit() no ha sido llamado.
            ValueError: Si metric no es válido.
        """
        self._check_fitted()

        valid_metrics = {"confidence", "lift", "support", "leverage", "conviction"}
        if metric not in valid_metrics:
            raise ValueError(
                f"metric debe ser uno de {valid_metrics}. Recibido: '{metric}'"
            )

        from mlxtend.frequent_patterns import association_rules

        # Recalcular itemsets si no existen o si se pide un soporte más bajo
        if self._itemsets is None or min_support < self._last_min_support:
            self.get_frequent_itemsets(min_support=min_support)

        if self._itemsets is None or self._itemsets.empty:
            self._rules = pd.DataFrame()
            return self._rules

        # El threshold de mlxtend debe corresponder a la métrica elegida
        metric_threshold_map: dict[str, float] = {
            "confidence": min_confidence,
            "lift":       min_lift,
            "support":    min_support,
            "leverage":   0.0,
            "conviction": 1.0,
        }
        threshold = metric_threshold_map[metric]

        rules = association_rules(
            self._itemsets,
            metric=metric,
            min_threshold=threshold,
        )

        if rules.empty:
            self._rules = pd.DataFrame()
            return self._rules

        # Filtros post-generación (independientes de la métrica principal)
        rules = rules[
            (rules["confidence"] >= min_confidence) &
            (rules["lift"]       >= min_lift)
        ].copy()

        # Columnas legibles para UI
        rules["antecedents_str"] = rules["antecedents"].apply(_frozenset_to_str)
        rules["consequents_str"] = rules["consequents"].apply(_frozenset_to_str)

        # Redondear numéricas; conviction puede ser inf cuando confidence=1.0
        numeric_cols = ["support", "confidence", "lift", "leverage", "conviction"]
        for col in (c for c in numeric_cols if c in rules.columns):
            rules[col] = (
                pd.to_numeric(rules[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .round(4)
            )

        self._rules = (
            rules
            .sort_values("lift", ascending=False)
            .reset_index(drop=True)
        )
        return self._rules

    # ── Motor de recomendaciones ──────────────────────────────

    def recommend(
        self,
        item: str,
        top_n: int = 5,
        sort_by: Literal["confidence", "lift", "support"] = "confidence",
    ) -> pd.DataFrame:
        """
        Recomienda los consecuentes más probables dado un ítem antecedente.

        Args:
            item:    Nombre del ítem (debe existir en las transacciones).
            top_n:   Número máximo de recomendaciones.
            sort_by: Métrica de ordenamiento ('confidence' | 'lift' | 'support').

        Returns:
            DataFrame con columnas: Recomendación, Confianza, Lift, Soporte.
            DataFrame vacío si no hay reglas que contengan el ítem.

        Raises:
            RuntimeError: Si get_rules() no ha sido llamado previamente.
        """
        self._check_rules()

        mask = self._rules["antecedents"].apply(lambda x: item in x)
        subset = self._rules.loc[mask].sort_values(sort_by, ascending=False).head(top_n)

        if subset.empty:
            return pd.DataFrame(
                columns=["Recomendación", "Confianza", "Lift", "Soporte"]
            )

        return pd.DataFrame({
            "Recomendación": subset["consequents_str"].values,
            "Confianza":     subset["confidence"].values,
            "Lift":          subset["lift"].values,
            "Soporte":       subset["support"].values,
        })

    # ── Estadísticas y resumen ────────────────────────────────

    def summary(self) -> dict:
        """
        Resumen estadístico del análisis (disponible aunque no haya reglas).

        Returns:
            Dict con conteos de transacciones, ítems únicos, itemsets y reglas,
            más estadísticas descriptivas de confianza/lift si hay reglas.
        """
        result: dict = {
            "total_transacciones":  len(self._transacciones) if self._transacciones else 0,
            "total_items_unicos":   self._encoded_df.shape[1] if self._encoded_df is not None else 0,
            "itemsets_frecuentes":  len(self._itemsets) if self._itemsets is not None else 0,
            "reglas_generadas":     0,
            "confianza_media":      None,
            "confianza_max":        None,
            "lift_medio":           None,
            "lift_max":             None,
            "soporte_medio":        None,
        }

        if self._rules is not None and not self._rules.empty:
            r = self._rules
            result.update({
                "reglas_generadas": len(r),
                "confianza_media":  round(float(r["confidence"].mean()), 4),
                "confianza_max":    round(float(r["confidence"].max()), 4),
                "lift_medio":       round(float(r["lift"].mean()), 4),
                "lift_max":         round(float(r["lift"].max()), 4),
                "soporte_medio":    round(float(r["support"].mean()), 4),
            })

        return result

    def top_itemsets(self, n: int = 10, min_size: int = 2) -> pd.DataFrame:
        """
        Retorna los n itemsets con mayor soporte y al menos min_size ítems.

        Args:
            n:        Número máximo de itemsets a retornar.
            min_size: Tamaño mínimo del itemset.

        Returns:
            DataFrame con columnas: itemsets_str, support, n_items.
        """
        if self._itemsets is None or self._itemsets.empty:
            return pd.DataFrame(columns=["itemsets_str", "support", "n_items"])

        filtered = self._itemsets[self._itemsets["n_items"] >= min_size]
        return (
            filtered[["itemsets_str", "support", "n_items"]]
            .sort_values("support", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def top_rules(
        self,
        n: int = 10,
        by: Literal["lift", "confidence", "support"] = "lift",
    ) -> pd.DataFrame:
        """
        Retorna las n mejores reglas ordenadas por la métrica indicada.

        Args:
            n:  Número máximo de reglas a retornar.
            by: Columna de ordenamiento ('lift' | 'confidence' | 'support').

        Returns:
            DataFrame con las columnas más relevantes para presentación.
        """
        if self._rules is None or self._rules.empty:
            return pd.DataFrame()

        display_cols = [
            "antecedents_str", "consequents_str",
            "support", "confidence", "lift", "conviction",
        ]
        available = [c for c in display_cols if c in self._rules.columns]
        return (
            self._rules[available]
            .sort_values(by, ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def filter_rules_by_item(
        self,
        item: str,
        side: _SIDE_LITERAL = "antecedents",
    ) -> pd.DataFrame:
        """
        Filtra reglas que contienen un ítem específico.

        Args:
            item: String del ítem a buscar.
            side: Dónde buscar — 'antecedents' | 'consequents' | 'any'.

        Returns:
            DataFrame filtrado con columnas principales.

        Raises:
            ValueError: Si side no es un valor válido.
            RuntimeError: Si get_rules() no ha sido llamado.
        """
        self._check_rules()

        valid_sides = ("antecedents", "consequents", "any")
        if side not in valid_sides:
            raise ValueError(
                f"side debe ser uno de {valid_sides}. Recibido: '{side}'"
            )

        if self._rules.empty:
            return pd.DataFrame()

        in_ante = self._rules["antecedents"].apply(lambda x: item in x)
        in_cons = self._rules["consequents"].apply(lambda x: item in x)

        mask_map = {
            "antecedents": in_ante,
            "consequents": in_cons,
            "any":         in_ante | in_cons,
        }
        mask = mask_map[side]

        display_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
        available = [c for c in display_cols if c in self._rules.columns]
        return self._rules.loc[mask, available].reset_index(drop=True)

    # ── Distribución de ítems ─────────────────────────────────

    def distribution_stats(self) -> dict:
        """
        Estadísticas de la distribución de ítems por transacción.

        Returns:
            Dict vacío si fit() no ha sido llamado.
        """
        if not self._fitted or self._transacciones is None:
            return {}

        sizes = pd.Series([len(t) for t in self._transacciones])
        return {
            "media_items_por_transaccion": round(float(sizes.mean()), 2),
            "mediana_items":               round(float(sizes.median()), 2),
            "max_items":                   int(sizes.max()),
            "min_items":                   int(sizes.min()),
            "transacciones_1_item":        int((sizes == 1).sum()),
            "transacciones_5_mas_items":   int((sizes >= 5).sum()),
        }
