"""
web_mining.py  ·  Módulo de Web Mining genérico
BCD-7213 Minería de Datos Avanzada - Universidad LEAD

MEJORA: WebMiner soporta URLs genéricas (no solo la tienda hardcoded).
  - load_from_url(): carga directo desde CSV/JSON/Excel/HTML-table en cualquier URL
  - scrape_with_log(): acepta url_base + css_selector genérico + field_map
  - Detecta automáticamente si la URL es un archivo de datos o una página para scraping
  - Fallback sintético y delay conservados
"""

import re
import time
import io
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# MEJORA: constantes ahora son defaults, no absolutos hardcoded
_DEFAULT_BASE_URL  = "https://tiendadeaventuracr.com"
_DEFAULT_COLL_PATH = "/collections/todos-nuestros-productos"
_DEFAULT_SELECTOR  = "div.productitem"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-CR,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class WebMiner:
    """
    Motor de Web Mining genérico para e-commerce y fuentes de datos web.

    MEJORA: ya no está atado a una sola tienda. Acepta cualquier URL base,
    selector CSS para scraping, o URL directa a CSV/JSON/Excel.

    Métodos principales
    -------------------
    load_from_url(url)                        → DataFrame   # MEJORA: nuevo
    scrape_with_log(url_base, css_selector,   → (DataFrame, list[str])
                    max_pages, field_map)      # MEJORA: firma extendida
    get_fallback_dataset()                    → DataFrame
    summary_stats(df)                         → dict
    extract_text(url, tag, css_class)         → list[str]
    regex_extract(text, pattern, flags)       → list[str]
    regex_clean(text, pattern, replacement)   → str
    """

    def __init__(self, delay: float = 1.0, timeout: int = 10):
        self.delay   = delay
        self.timeout = timeout
        self._session = self._init_session()

    def _init_session(self):
        try:
            import requests
            s = requests.Session()
            s.headers.update(HEADERS)
            return s
        except ImportError:
            return None

    # ── Petición HTTP ──────────────────────────────────────────

    def _get(self, url: str):
        """GET con manejo de errores. Retorna BeautifulSoup o None."""
        if self._session is None:
            return None
        try:
            from bs4 import BeautifulSoup
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return None

    # ── MEJORA: carga directa desde URL (CSV/JSON/Excel/HTML table) ──────────

    def load_from_url(self, url: str) -> pd.DataFrame:
        """
        MEJORA: Carga un DataFrame directamente desde una URL pública.
        Soporta: CSV, TSV, JSON, Excel (.xlsx/.xls), páginas HTML con <table>.
        Útil para integrar datasets públicos sin scraping manual.

        Args:
            url: URL directa al archivo de datos o página con tabla HTML.

        Returns:
            pd.DataFrame listo para benchmarking, o DataFrame vacío si falla.
        """
        url_lower = url.lower().split("?")[0]

        # MEJORA: intentar CSV (con varios separadores)
        if any(url_lower.endswith(ext) for ext in (".csv", ".tsv", ".txt")):
            sep = "\t" if url_lower.endswith(".tsv") else ","
            for s in [sep, ";", "|"]:
                try:
                    df = pd.read_csv(url, sep=s)
                    if not df.empty and len(df.columns) > 1:
                        return df
                except Exception:
                    pass

        # MEJORA: intentar JSON
        if url_lower.endswith(".json"):
            try:
                df = pd.read_json(url)
                if not df.empty:
                    return df
            except Exception:
                pass

        # MEJORA: intentar Excel (necesita descargar bytes primero)
        if any(url_lower.endswith(ext) for ext in (".xlsx", ".xls")):
            if self._session:
                try:
                    resp = self._session.get(url, timeout=self.timeout)
                    resp.raise_for_status()
                    df = pd.read_excel(io.BytesIO(resp.content))
                    if not df.empty:
                        return df
                except Exception:
                    pass

        # MEJORA: catch-all — pandas.read_csv maneja URLs de GitHub raw, etc.
        try:
            df = pd.read_csv(url)
            if not df.empty and len(df.columns) > 1:
                return df
        except Exception:
            pass

        # MEJORA: fallback a tabla HTML en la página
        try:
            tables = pd.read_html(url)
            if tables:
                df = max(tables, key=len)
                if not df.empty:
                    return df
        except Exception:
            pass

        return pd.DataFrame()

    # ── MEJORA: parser genérico con field_map ──────────────────

    def _parse_generic_row(self, tag, field_map: dict) -> dict | None:
        """
        MEJORA: Extrae campos de cualquier tag HTML dado un dict de selectores.
        field_map = {"NombreColumna": "selector_css", ...}
        Si field_map está vacío, extrae todo el texto como columna "Texto".
        """
        if not field_map:
            text = tag.get_text(separator=" ", strip=True)
            return {"Texto": text} if text else None

        row = {}
        for col_name, selector in field_map.items():
            try:
                el = tag.select_one(selector) if selector else tag
                row[col_name] = el.get_text(strip=True) if el else None
            except Exception:
                row[col_name] = None
        return row if any(v is not None for v in row.values()) else None

    # ── Parser de tienda de aventura (compatibilidad) ──────────

    def _parse_product(self, tag) -> dict | None:
        """Extrae campos del producto de la tienda hardcoded (compatibilidad)."""
        nombre = precio_orig = precio_desc = None
        agotado = False

        h2 = tag.find("h2", {"class": "productitem--title"})
        if h2 and h2.find("a"):
            nombre = h2.find("a").text.strip()

        po = tag.find("div", {"class": "price__current price__current--emphasize"})
        if po:
            precio_orig = po.get_text(" ", strip=True)

        pd_tag = tag.find(
            "div",
            {"class": "price__current price__current--emphasize price__current--on-sale"})
        if pd_tag:
            precio_desc = pd_tag.get_text(" ", strip=True)
            base = tag.find("span", {"class": "money price__compare-at--single"})
            if base:
                precio_orig = base.get_text(" ", strip=True)

        if tag.find(class_=re.compile(r"soldout|sold.out|agotado", re.I)):
            agotado = True

        if nombre is None:
            return None

        return {
            "Nombre":          nombre,
            "PrecioOriginal":  precio_orig,
            "PrecioDescuento": precio_desc,
            "Agotado":         "Sí" if agotado else "No",
        }

    # ── Limpieza de precio ─────────────────────────────────────

    @staticmethod
    def _clean_price(val) -> float | None:
        if val is None:
            return None
        s = str(val)
        s = re.sub(r"(?i)(precio\s+(original|actual|base)|price)", "", s)
        s = re.sub(r"[$,\n\r\t\xa0]", "", s)
        nums = re.findall(r"\d+\.?\d*", s)
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                pass
        return None

    # ── MEJORA: scrape_with_log con firma extendida ────────────

    def scrape_with_log(
        self,
        url_base: str | None = None,
        css_selector: str | None = None,
        max_pages: int = 3,
        field_map: dict | None = None,
    ) -> tuple:
        """
        MEJORA: Extrae datos de cualquier URL, no solo la tienda hardcoded.

        Casos soportados:
          1. URL directa a CSV/JSON/Excel  → carga sin scraping (load_from_url)
          2. URL + css_selector            → scraping con parser genérico
          3. Sin argumentos                → tienda de aventura (compatibilidad)

        Args:
            url_base     : URL base (o URL directa a datos).
                           None → usa tienda de aventura CR.
            css_selector : Selector CSS para los items (ej. "article.product").
                           None → usa selector de la tienda de aventura.
            max_pages    : Páginas máximas a paginar.
            field_map    : {"Columna": "selector_css"} para parser genérico.
                           None → usa parser de producto de la tienda.

        Returns:
            (DataFrame, list[str])
        """
        log  = []
        rows = []

        effective_url = url_base or (_DEFAULT_BASE_URL + _DEFAULT_COLL_PATH)
        effective_sel = css_selector or _DEFAULT_SELECTOR
        url_lower     = effective_url.lower().split("?")[0]

        # MEJORA: detectar URL directa a archivo de datos
        is_data_url = any(url_lower.endswith(ext)
                          for ext in (".csv", ".tsv", ".json", ".xlsx", ".xls"))
        # MEJORA: también detectar URLs típicas de datasets (GitHub raw, etc.)
        is_data_url = is_data_url or any(
            kw in effective_url
            for kw in ["raw.githubusercontent", "/download/", "data.csv",
                        "dataset", "kaggle.com/api"])

        if is_data_url:
            log.append(f"📥 URL de datos detectada: {effective_url}")
            df = self.load_from_url(effective_url)
            if not df.empty:
                log.append(f"✅ {len(df)} filas · {len(df.columns)} columnas")
                return df, log
            log.append("⚠️ No se pudo cargar — usando dataset sintético")
            return self.get_fallback_dataset(), log

        # MEJORA: paginación adaptativa según formato de la URL
        has_page_param = "page=" in effective_url

        for page in range(1, max_pages + 1):
            # MEJORA: construir URL de página según formato detectado
            if has_page_param:
                url = re.sub(r"page=\d+", f"page={page}", effective_url)
            elif url_base and url_base not in (_DEFAULT_BASE_URL + _DEFAULT_COLL_PATH):
                sep = "&" if "?" in effective_url else "?"
                url = f"{effective_url}{sep}page={page}"
            else:
                url = f"{_DEFAULT_BASE_URL}{_DEFAULT_COLL_PATH}?page={page}"

            log.append(f"📄 Página {page}: {url}")
            soup = self._get(url)

            if soup is None:
                log.append(f"  ❌ Sin respuesta en página {page}")
                break

            # MEJORA: parser genérico si se proporcionó selector o field_map
            if field_map is not None or (css_selector and css_selector != _DEFAULT_SELECTOR):
                items = soup.select(effective_sel)
                if not items:
                    log.append(f"  ⏹️ '{effective_sel}' no encontró items en pág {page}")
                    break
                for item in items:
                    row = self._parse_generic_row(item, field_map or {})
                    if row:
                        rows.append(row)
                log.append(f"  ✅ {len(items)} items con selector '{effective_sel}'")
            else:
                # Compatibilidad: parser de la tienda de aventura
                products = soup.find_all("div", {"class": "productitem"})
                if not products:
                    log.append(f"  ⏹️ No hay más productos en página {page}")
                    break
                for prod in products:
                    data = self._parse_product(prod)
                    if data:
                        rows.append(data)
                log.append(f"  ✅ {len(products)} productos extraídos")

            if page < max_pages:
                time.sleep(self.delay)

        if not rows:
            log.append("⚠️ Sin datos reales — usando dataset sintético")
            return self.get_fallback_dataset(), log

        df = pd.DataFrame(rows)

        if "PrecioOriginal" in df.columns:
            df = self._build_features(df)
            log.append(f"🧹 Limpieza completada — {len(df)} productos válidos")
        else:
            # MEJORA: conversión numérica genérica para cualquier scraping
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            log.append(f"✅ {len(df)} filas extraídas")

        return df, log

    # ── Construcción de features (tienda) ──────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["PrecioOriginal"]  = df["PrecioOriginal"].apply(self._clean_price)
        df["PrecioDescuento"] = df["PrecioDescuento"].apply(self._clean_price)
        df["PrecioFinal"]     = df["PrecioDescuento"].combine_first(df["PrecioOriginal"])
        df["TieneDescuento"]  = df["PrecioDescuento"].notna().astype(int)
        df["PctDescuento"]    = 0.0
        mask = df["TieneDescuento"] == 1
        if mask.any():
            df.loc[mask, "PctDescuento"] = (
                (df.loc[mask, "PrecioOriginal"] - df.loc[mask, "PrecioDescuento"])
                / df.loc[mask, "PrecioOriginal"].replace(0, np.nan) * 100
            ).round(2).fillna(0)
        return df.dropna(subset=["PrecioOriginal"]).reset_index(drop=True)

    # ── Dataset sintético ──────────────────────────────────────

    @staticmethod
    def get_fallback_dataset() -> pd.DataFrame:
        """200 productos sintéticos de aventura outdoor. Siempre disponible."""
        rng = np.random.RandomState(42)
        categorias = ["Sombrero","Gorra","Mochila","Carpa","Botas",
                      "Linterna","Navaja","Brújula","Cantimplora","Cuerda",
                      "Guantes","Chaleco","Calcetines","Bastones","Sleeping Bag",
                      "Malla Protectora","Lentes UV","Reloj GPS","Carabinero","Casco"]
        marcas = ["Sunday Afternoons","NRS","Black Diamond","Osprey","Salomon",
                  "Petzl","Leatherman","Silva","Nalgene","Mammut",
                  "Patagonia","Arc'teryx","MSR","Sea to Summit","Garmin"]
        rangos = {
            "Sombrero":(25,90),"Gorra":(20,65),"Mochila":(80,280),"Carpa":(150,600),
            "Botas":(100,400),"Linterna":(18,120),"Navaja":(30,180),"Brújula":(15,70),
            "Cantimplora":(15,60),"Cuerda":(40,220),"Guantes":(20,90),"Chaleco":(60,220),
            "Calcetines":(10,35),"Bastones":(40,180),"Sleeping Bag":(80,350),
            "Malla Protectora":(15,45),"Lentes UV":(30,180),"Reloj GPS":(150,500),
            "Carabinero":(10,60),"Casco":(60,250),
        }
        n    = 200
        cats = rng.choice(categorias, n)
        marc = rng.choice(marcas, n)
        noms = [f"{marc[i]} {cats[i]} {rng.choice(['Pro','Elite','Sport','Lite','Ultra'])} "
                f"{rng.randint(100,999)}" for i in range(n)]
        precios     = np.array([rng.uniform(*rangos[c]) for c in cats]).round(2)
        tiene_desc  = rng.random(n) < 0.28
        desc_pct    = np.where(tiene_desc, rng.uniform(10, 45, n).round(1), 0.0)
        precio_desc = np.where(tiene_desc, (precios*(1-desc_pct/100)).round(2), np.nan)
        agotado     = np.where(rng.random(n) < 0.06, "Sí", "No")
        df = pd.DataFrame({
            "Nombre":          noms,
            "Categoria":       cats,
            "Marca":           marc,
            "PrecioOriginal":  precios,
            "PrecioDescuento": precio_desc,
            "Agotado":         agotado,
        })
        df["PrecioFinal"]    = df["PrecioDescuento"].combine_first(df["PrecioOriginal"])
        df["TieneDescuento"] = tiene_desc.astype(int)
        df["PctDescuento"]   = desc_pct
        return df

    # ── Extracción genérica ────────────────────────────────────

    def extract_text(self, url: str, tag: str = "p",
                     css_class: str | None = None) -> list:
        """Extrae textos de una etiqueta HTML desde una URL."""
        soup = self._get(url)
        if soup is None:
            return []
        attrs = {"class": css_class} if css_class else {}
        elems = soup.find_all(tag, attrs) if attrs else soup.find_all(tag)
        return [e.get_text(strip=True) for e in elems if e.get_text(strip=True)]

    @staticmethod
    def regex_extract(text: str, pattern: str, flags: int = 0) -> list:
        """Aplica regex y retorna todas las coincidencias."""
        try:
            return re.findall(pattern, text, flags)
        except re.error:
            return []

    @staticmethod
    def regex_clean(text: str, pattern: str, replacement: str = "") -> str:
        """Elimina o reemplaza un patrón en el texto."""
        try:
            return re.sub(pattern, replacement, text).strip()
        except re.error:
            return text

    # ── Estadísticas ───────────────────────────────────────────

    @staticmethod
    def summary_stats(df: pd.DataFrame) -> dict:
        """Resumen estadístico del dataset."""
        num_cols = df.select_dtypes(include="number").columns
        col = ("PrecioFinal" if "PrecioFinal" in df.columns
               else num_cols[0] if len(num_cols) else None)
        n = len(df)
        result = {
            "total_productos":   n,
            "con_descuento":     int(df.get("TieneDescuento", pd.Series(0)).sum()),
            "agotados":          int((df.get("Agotado", pd.Series("No")) == "Sí").sum()),
            "tasa_descuento_%":  round(
                int(df.get("TieneDescuento", pd.Series(0)).sum()) / n * 100, 1) if n else 0,
        }
        if col:
            result.update({
                "precio_promedio": round(float(df[col].mean()), 2),
                "precio_mediano":  round(float(df[col].median()), 2),
                "precio_maximo":   round(float(df[col].max()), 2),
                "precio_minimo":   round(float(df[col].min()), 2),
                "precio_std":      round(float(df[col].std()), 2),
                "ahorro_promedio": round(float(
                    (df["PrecioOriginal"] - df[col]).mean()), 2)
                    if "PrecioOriginal" in df.columns and col != "PrecioOriginal" else 0,
            })
        return result