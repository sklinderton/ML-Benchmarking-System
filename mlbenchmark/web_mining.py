"""
web_mining.py  ·  Módulo de Web Mining REAL
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
Caso de Estudio 2  ·  I Cuatrimestre 2026

Web scraping real con BeautifulSoup + requests:
  · Paginación automática
  · Headers de navegador real (evita bloqueos básicos)
  · Logging de progreso por página
  · Limpieza robusta con expresiones regulares
  · Fallback a dataset sintético cuando no hay conexión
"""

import re
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_URL  = "https://tiendadeaventuracr.com"
COLL_PATH = "/collections/todos-nuestros-productos"
HEADERS   = {
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
    Motor de Web Mining para tiendas e-commerce.

    Métodos principales
    -------------------
    scrape_with_log(max_pages) → (DataFrame, list[str])
    get_fallback_dataset()     → DataFrame
    summary_stats(df)          → dict
    extract_text(url, tag)     → list[str]
    regex_extract(text, pat)   → list[str]
    regex_clean(text, pat)     → str
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
        """Realiza GET con manejo de errores. Retorna BeautifulSoup o None."""
        if self._session is None:
            return None
        try:
            from bs4 import BeautifulSoup
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return None

    # ── Parsing de un producto ─────────────────────────────────

    def _parse_product(self, tag) -> dict | None:
        """Extrae campos de un div.productitem."""
        nombre = precio_orig = precio_desc = None
        agotado = False

        h2 = tag.find("h2", {"class": "productitem--title"})
        if h2 and h2.find("a"):
            nombre = h2.find("a").text.strip()

        # Precio base
        po = tag.find("div", {"class": "price__current price__current--emphasize"})
        if po:
            precio_orig = po.get_text(" ", strip=True)

        # Precio con descuento
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
        # Elimina texto dinámico, símbolos y espacios
        s = re.sub(r"(?i)(precio\s+(original|actual|base)|price)", "", s)
        s = re.sub(r"[$,\n\r\t\xa0]", "", s)
        # Extrae primer número válido
        nums = re.findall(r"\d+\.?\d*", s)
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                pass
        return None

    # ── Scraping con paginación ────────────────────────────────

    def scrape_with_log(self, max_pages: int = 3) -> tuple[pd.DataFrame, list]:
        """
        Extrae productos paginando hasta max_pages.

        Returns
        -------
        (DataFrame limpio, lista de mensajes de log)
        """
        rows = []
        log  = []

        for page in range(1, max_pages + 1):
            url  = f"{BASE_URL}{COLL_PATH}?page={page}"
            log.append(f"📄 Página {page}: {url}")
            soup = self._get(url)

            if soup is None:
                log.append(f"  ❌ Sin respuesta en página {page}")
                break

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
            log.append("⚠️ Sin datos reales — cargando dataset sintético")
            return self.get_fallback_dataset(), log

        df = pd.DataFrame(rows)
        df = self._build_features(df)
        log.append(f"🧹 Limpieza completada — {len(df)} productos válidos")
        return df, log

    # ── Construcción de features ───────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["PrecioOriginal"]  = df["PrecioOriginal"].apply(self._clean_price)
        df["PrecioDescuento"] = df["PrecioDescuento"].apply(self._clean_price)

        df["PrecioFinal"]     = df["PrecioDescuento"].combine_first(df["PrecioOriginal"])
        df["TieneDescuento"]  = df["PrecioDescuento"].notna().astype(int)

        df["PctDescuento"] = 0.0
        mask = df["TieneDescuento"] == 1
        if mask.any():
            df.loc[mask, "PctDescuento"] = (
                (df.loc[mask, "PrecioOriginal"] - df.loc[mask, "PrecioDescuento"])
                / df.loc[mask, "PrecioOriginal"].replace(0, np.nan) * 100
            ).round(2).fillna(0)

        df = df.dropna(subset=["PrecioOriginal"]).reset_index(drop=True)
        return df

    # ── Dataset sintético ──────────────────────────────────────

    @staticmethod
    def get_fallback_dataset() -> pd.DataFrame:
        """
        Dataset sintético de 200 productos de aventura outdoor.
        Mismas columnas que el scraping real — siempre disponible offline.
        """
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

        precios = np.array([rng.uniform(*rangos[c]) for c in cats]).round(2)

        tiene_desc = rng.random(n) < 0.28
        desc_pct   = np.where(tiene_desc, rng.uniform(10, 45, n).round(1), 0.0)
        precio_desc = np.where(
            tiene_desc, (precios * (1 - desc_pct / 100)).round(2), np.nan)

        agotado = np.where(rng.random(n) < 0.06, "Sí", "No")

        df = pd.DataFrame({
            "Nombre":          noms,
            "Categoria":       cats,
            "Marca":           marc,
            "PrecioOriginal":  precios,
            "PrecioDescuento": precio_desc,
            "Agotado":         agotado,
        })

        df["PrecioFinal"]   = df["PrecioDescuento"].combine_first(df["PrecioOriginal"])
        df["TieneDescuento"]= tiene_desc.astype(int)
        df["PctDescuento"]  = desc_pct

        return df

    # ── Extracción genérica de texto ───────────────────────────

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
        col = "PrecioFinal" if "PrecioFinal" in df.columns else "PrecioOriginal"
        n   = len(df)
        return {
            "total_productos":   n,
            "con_descuento":     int(df.get("TieneDescuento", pd.Series(0)).sum()),
            "agotados":          int((df.get("Agotado", pd.Series("No")) == "Sí").sum()),
            "precio_promedio":   round(float(df[col].mean()), 2),
            "precio_mediano":    round(float(df[col].median()), 2),
            "precio_maximo":     round(float(df[col].max()), 2),
            "precio_minimo":     round(float(df[col].min()), 2),
            "precio_std":        round(float(df[col].std()), 2),
            "tasa_descuento_%":  round(int(df.get("TieneDescuento",
                                                    pd.Series(0)).sum()) / n * 100, 1) if n else 0,
            "ahorro_promedio":   round(float(
                (df["PrecioOriginal"] - df["PrecioFinal"]).mean()), 2)
                if "PrecioFinal" in df.columns else 0,
        }