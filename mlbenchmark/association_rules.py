"""
association_rules.py  ·  Reglas de Asociación (Apriori + FP-Growth)
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
  · Visualizaciones: scatter support×confidence×lift, heatmap, barras

Función:
    generate_synthetic_transactions(productos_df) → list[list[str]]
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# DATASET FALLBACK (transacciones de comestibles)
# ──────────────────────────────────────────────────────────────
GROCERIES_URL = (
    "https://raw.githubusercontent.com/JoaquinAmatRodrigo/"
    "Estadistica-con-R/master/datos/datos_groceries.csv"
)


def load_groceries_dataset() -> pd.DataFrame:
    """
    Carga el dataset público de comestibles (groceries).
    Columnas: id_compra, item
    Fallback: genera transacciones sintéticas si no hay conexión.
    """
    try:
        df = pd.read_csv(GROCERIES_URL)
        if "id_compra" in df.columns and "item" in df.columns:
            return df
    except Exception:
        pass
    return _synthetic_groceries()


def _synthetic_groceries() -> pd.DataFrame:
    """Transacciones sintéticas de productos de aventura outdoor."""
    rng = np.random.RandomState(42)
    categorias = [
        "Sombrero", "Mochila", "Botas", "Linterna", "Cantimplora",
        "Guantes", "Chaleco", "Sleeping Bag", "Bastones", "Navaja",
        "Brújula", "Cuerda", "Calcetines", "Carpa", "Gorra",
    ]
    # Frecuencias realistas (algunos productos se compran más juntos)
    cooccurrence = {
        "Mochila":      ["Cantimplora", "Linterna", "Botas"],
        "Botas":        ["Calcetines", "Bastones"],
        "Carpa":        ["Sleeping Bag", "Linterna", "Mochila"],
        "Sleeping Bag": ["Carpa", "Linterna"],
        "Linterna":     ["Pilas", "Mochila"],
        "Guantes":      ["Chaleco", "Sombrero"],
    }

    rows = []
    for compra_id in range(1, 2001):
        n_items = rng.randint(1, 7)
        items = list(rng.choice(categorias, size=min(n_items, len(categorias)),
                                replace=False))
        # Añadir co-ocurrencias
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
) -> list:
    """
    Genera transacciones sintéticas a partir de un DataFrame de productos.

    Simula un escenario de market basket analysis usando las categorías
    del dataset de Web Mining.

    Parameters
    ----------
    productos_df         : DataFrame con columna 'Categoria' (o 'Nombre')
    n_transacciones      : número de transacciones a generar
    max_items_por_compra : máximo de productos por transacción
    seed                 : semilla aleatoria

    Returns
    -------
    list de listas de strings (cada lista es una transacción)
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

    transacciones = []
    for _ in range(n_transacciones):
        n_items = rng.randint(1, min(max_items_por_compra + 1, len(items_pool) + 1))
        compra  = list(rng.choice(items_pool, size=n_items, replace=False))
        transacciones.append(compra)

    return transacciones


# ──────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ──────────────────────────────────────────────────────────────

class AssociationRulesMiner:
    """
    Motor de Reglas de Asociación basado en el algoritmo Apriori.

    Workflow
    --------
    1. miner = AssociationRulesMiner()
    2. miner.fit(transacciones)               ← lista de listas de items
    3. df_itemsets = miner.get_frequent_itemsets(min_support=0.02)
    4. df_rules    = miner.get_rules(min_confidence=0.3, min_lift=1.0)
    5. recomend    = miner.recommend("Mochila", top_n=5)
    """

    def __init__(self):
        self._encoder         = None
        self._encoded_df      = None
        self._transacciones   = None
        self._itemsets        = None
        self._rules           = None
        self._fitted          = False

    # ── Entrenamiento ─────────────────────────────────────────

    def fit(self, transacciones: list) -> "AssociationRulesMiner":
        """
        Codifica las transacciones en matriz binaria y entrena el encoder.

        Parameters
        ----------
        transacciones : list de listas de strings
                        ej. [["Mochila","Botas"], ["Linterna","Carpa"], …]
        """
        try:
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            raise ImportError("Instala mlxtend:  pip install mlxtend")

        self._transacciones = transacciones
        te = TransactionEncoder()
        matrix = te.fit(transacciones).transform(transacciones)
        # Force plain Python str column names — mlxtend 0.24 produces np.str_
        # which breaks association_rules() due to a numpy vectorize bug.
        self._encoded_df = pd.DataFrame(matrix, columns=[str(c) for c in te.columns_])
        self._encoder    = te
        self._fitted     = True
        return self

    def fit_from_dataframe(self, df: pd.DataFrame,
                            id_col: str = "id_compra",
                            item_col: str = "item") -> "AssociationRulesMiner":
        """
        Construye transacciones directamente desde un DataFrame largo
        (una fila = un ítem de una compra).
        """
        transacciones = (
            df.groupby(id_col)[item_col]
              .apply(list)
              .tolist()
        )
        return self.fit(transacciones)

    # ── Itemsets frecuentes ───────────────────────────────────

    def get_frequent_itemsets(self, min_support: float = 0.02,
                               max_len: int | None = None) -> pd.DataFrame:
        """
        Identifica itemsets frecuentes con soporte ≥ min_support.

        Parameters
        ----------
        min_support : fracción mínima de transacciones (0.0–1.0)
        max_len     : longitud máxima del itemset (None = sin límite)

        Returns
        -------
        pd.DataFrame con columnas: support, itemsets, n_items
        """
        if not self._fitted:
            raise RuntimeError("Llama .fit() primero.")
        try:
            from mlxtend.frequent_patterns import apriori
        except ImportError:
            raise ImportError("Instala mlxtend:  pip install mlxtend")

        kw = {"min_support": min_support, "use_colnames": True}
        if max_len:
            kw["max_len"] = max_len

        self._itemsets = apriori(self._encoded_df, **kw)
        # Fix np.str_ items in frozensets (mlxtend 0.24 bug with numpy vectorize)
        self._itemsets["itemsets"] = self._itemsets["itemsets"].apply(
            lambda fs: frozenset(str(x) for x in fs))
        self._itemsets["n_items"] = self._itemsets["itemsets"].apply(len)
        return self._itemsets.sort_values("support", ascending=False).reset_index(drop=True)

    # ── Reglas de asociación ──────────────────────────────────

    def get_rules(self, min_confidence: float = 0.3,
                  min_lift: float = 1.0,
                  min_support: float = 0.01,
                  metric: str = "confidence") -> pd.DataFrame:
        """
        Genera reglas de asociación a partir de los itemsets frecuentes.

        Parameters
        ----------
        min_confidence : confianza mínima (0–1)
        min_lift       : lift mínimo (> 1 indica asociación positiva)
        min_support    : recalcula itemsets si no han sido calculados
        metric         : métrica principal ('confidence' | 'lift' | 'support')

        Returns
        -------
        pd.DataFrame con columnas:
            antecedents, consequents, support, confidence, lift,
            leverage, conviction, antecedents_str, consequents_str
        """
        try:
            from mlxtend.frequent_patterns import association_rules
        except ImportError:
            raise ImportError("Instala mlxtend:  pip install mlxtend")

        if self._itemsets is None:
            self.get_frequent_itemsets(min_support=min_support)

        if self._itemsets.empty:
            return pd.DataFrame()

        rules = association_rules(
            self._itemsets, metric=metric, min_threshold=min_confidence)

        # Filtro adicional por lift
        rules = rules[rules["lift"] >= min_lift].copy()

        # Columnas legibles
        rules["antecedents_str"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(x)))

        # Redondear — conviction puede ser inf cuando confidence=1.0;
        # pd.to_numeric + replace inf → NaN garantiza dtype float antes de round
        for col in ["support", "confidence", "lift", "leverage", "conviction"]:
            if col in rules.columns:
                rules[col] = (pd.to_numeric(rules[col], errors="coerce")
                              .replace([np.inf, -np.inf], np.nan)
                              .round(4))

        self._rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        return self._rules

    # ── Motor de recomendaciones ──────────────────────────────

    def recommend(self, item: str, top_n: int = 5) -> pd.DataFrame:
        """
        Dado un ítem de antecedente, recomienda los consecuentes más probables.

        Parameters
        ----------
        item  : nombre del ítem (debe existir en las transacciones)
        top_n : número de recomendaciones a retornar

        Returns
        -------
        pd.DataFrame con: consequents_str, confidence, lift, support
        """
        if self._rules is None:
            raise RuntimeError("Genera reglas primero con .get_rules().")

        mask = self._rules["antecedents"].apply(lambda x: item in x)
        subset = self._rules.loc[mask].copy()

        if subset.empty:
            return pd.DataFrame(columns=["Recomendación", "Confianza", "Lift", "Soporte"])

        subset = subset.sort_values("confidence", ascending=False).head(top_n)
        return pd.DataFrame({
            "Recomendación": subset["consequents_str"].values,
            "Confianza":     subset["confidence"].values,
            "Lift":          subset["lift"].values,
            "Soporte":       subset["support"].values,
        })

    # ── Estadísticas ──────────────────────────────────────────

    def summary(self) -> dict:
        """Resumen estadístico de las reglas generadas."""
        if self._rules is None or self._rules.empty:
            return {}
        r = self._rules
        return {
            "total_transacciones":  len(self._transacciones) if self._transacciones else 0,
            "total_items_unicos":   self._encoded_df.shape[1] if self._encoded_df is not None else 0,
            "itemsets_frecuentes":  len(self._itemsets) if self._itemsets is not None else 0,
            "reglas_generadas":     len(r),
            "confianza_media":      round(float(r["confidence"].mean()), 4),
            "confianza_max":        round(float(r["confidence"].max()), 4),
            "lift_medio":           round(float(r["lift"].mean()), 4),
            "lift_max":             round(float(r["lift"].max()), 4),
            "soporte_medio":        round(float(r["support"].mean()), 4),
        }

    def top_itemsets(self, n: int = 10, min_size: int = 2) -> pd.DataFrame:
        """Retorna los n itemsets frecuentes con mayor soporte y al menos min_size ítems."""
        if self._itemsets is None:
            return pd.DataFrame()
        df = self._itemsets[self._itemsets["n_items"] >= min_size].copy()
        df["itemsets_str"] = df["itemsets"].apply(lambda x: " + ".join(sorted(x)))
        return df.sort_values("support", ascending=False).head(n)[
            ["itemsets_str", "support", "n_items"]
        ].reset_index(drop=True)

    def top_rules(self, n: int = 10,
                  by: str = "lift") -> pd.DataFrame:
        """Retorna las n mejores reglas ordenadas por la métrica indicada."""
        if self._rules is None or self._rules.empty:
            return pd.DataFrame()
        cols = ["antecedents_str", "consequents_str",
                "support", "confidence", "lift", "conviction"]
        available = [c for c in cols if c in self._rules.columns]
        return (self._rules[available]
                .sort_values(by, ascending=False)
                .head(n)
                .reset_index(drop=True))

    def filter_rules_by_item(self, item: str,
                              side: str = "antecedents") -> pd.DataFrame:
        """
        Filtra reglas que contienen un ítem en el antecedente o consecuente.

        Parameters
        ----------
        item : string del ítem a buscar
        side : 'antecedents' | 'consequents' | 'any'
        """
        if self._rules is None or self._rules.empty:
            return pd.DataFrame()
        if side == "antecedents":
            mask = self._rules["antecedents"].apply(lambda x: item in x)
        elif side == "consequents":
            mask = self._rules["consequents"].apply(lambda x: item in x)
        else:
            mask = (self._rules["antecedents"].apply(lambda x: item in x) |
                    self._rules["consequents"].apply(lambda x: item in x))
        cols = ["antecedents_str", "consequents_str",
                "support", "confidence", "lift"]
        available = [c for c in cols if c in self._rules.columns]
        return self._rules.loc[mask, available].reset_index(drop=True)

    # ── Distribuciones ────────────────────────────────────────

    def distribution_stats(self) -> dict:
        """Estadísticas de la distribución de ítems por transacción."""
        if self._transacciones is None:
            return {}
        sizes = pd.Series([len(t) for t in self._transacciones])
        return {
            "media_items_por_transaccion":  round(float(sizes.mean()), 2),
            "mediana_items":                round(float(sizes.median()), 2),
            "max_items":                    int(sizes.max()),
            "min_items":                    int(sizes.min()),
            "transacciones_1_item":         int((sizes == 1).sum()),
            "transacciones_5_mas_items":    int((sizes >= 5).sum()),
        }