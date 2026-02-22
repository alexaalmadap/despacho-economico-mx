import os
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Config
# ----------------------------
BASE_URL = "https://www.cenace.gob.mx"
LANDING_URL = f"{BASE_URL}/GraficaDemanda.aspx"
CENACE_URL = f"{BASE_URL}/GraficaDemanda.aspx/obtieneValoresTotal"

CACHE_DIR = "data_cache"  # Streamlit Cloud permite escribir aquí
os.makedirs(CACHE_DIR, exist_ok=True)

st.title("Demanda CENACE (Semana 2)")
st.write("Descarga por API de CENACE + batching (≤7 días) + cache en disco + checks de calidad.")

# ----------------------------
# Helpers
# ----------------------------

def _date_range_batches(start_dt: datetime, end_dt: datetime, batch_days: int = 7):
    """Genera rangos [start, end) en bloques de batch_days."""
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=batch_days), end_dt)
        yield cur, nxt
        cur = nxt


def _cache_path(system: str, start_dt: datetime, end_dt: datetime) -> str:
    """Ruta de cache por sistema y rango."""
    s = start_dt.strftime("%Y%m%d%H")
    e = end_dt.strftime("%Y%m%d%H")
    return os.path.join(CACHE_DIR, f"demanda_{system}_{s}_{e}.parquet")


def _parse_cenace_response(resp_json) -> pd.DataFrame:
    """
    Convierte la respuesta típica de servicios .aspx (a veces viene en resp_json['d'] como string JSON)
    a un DataFrame con columnas: timestamp, demand_mw
    """
    # Caso común ASP.NET: {"d":"[...]"} donde d es string con JSON adentro
    if isinstance(resp_json, dict) and "d" in resp_json:
        payload = resp_json["d"]
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload
    else:
        data = resp_json

    df = pd.DataFrame(data)

    # Encuentra columnas candidatas para tiempo
    col_time = None
    for c in df.columns:
        if str(c).lower() in ["fecha", "fechahora", "fecha_hora", "hora", "timestamp", "datetime", "fecha_hora_local"]:
            col_time = c
            break
    if col_time is None:
        # toma la primera columna tipo object como fallback
        for c in df.columns:
            if df[c].dtype == object:
                col_time = c
                break

    # Encuentra columnas candidatas para valor
    col_val = None
    for c in df.columns:
        if str(c).lower() in ["valor", "mw", "demanda", "demanda_mw", "value", "y"]:
            col_val = c
            break
    if col_val is None:
        # primera numérica
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                col_val = c
                break

    if col_time is None or col_val is None:
        raise ValueError(f"No pude identificar columnas tiempo/valor. Columnas: {list(df.columns)}")

    out = df[[col_time, col_val]].copy()
    out.columns = ["timestamp_raw", "demand_mw"]

    out["timestamp"] = pd.to_datetime(out["timestamp_raw"], errors="coerce", dayfirst=True)
    out["demand_mw"] = pd.to_numeric(out["demand_mw"], errors="coerce")

    out = out.drop(columns=["timestamp_raw"])
    out = out.dropna(subset=["timestamp", "demand_mw"]).sort_values("timestamp").reset_index(drop=True)

    return out


def fetch_cenace(system: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Descarga demanda de CENACE usando:
    1) GET inicial para obtener cookies
    2) POST con headers tipo navegador
    Devuelve DataFrame con columnas: timestamp, demand_mw
    """
    cache_file = _cache_path(system, start_dt, end_dt)
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            # si falla leer parquet, seguimos sin cache
            pass

    s = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
        "Referer": LANDING_URL,
        "Origin": BASE_URL,
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/json; charset=UTF-8",
    }

    # Paso 1: obtener cookies
    s.get(LANDING_URL, headers={"User-Agent": headers["User-Agent"]}, timeout=30)

    payload = {
        "sistema": system,
        "fechaInicio": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "fechaFin": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }

    r = s.post(CENACE_URL, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        st.error(f"CENACE respondió {r.status_code}")
        st.error(r.text[:300])
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    data = r.json()

    # Normaliza a (timestamp, demand_mw)
    try:
        df = _parse_cenace_response(data)
    except Exception as e:
        st.error("No pude parsear la respuesta de CENACE.")
        st.error(str(e))
        st.json(data if isinstance(data, dict) else {"data": str(type(data))})
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    # Filtra al rango por si viene extra
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].copy()

    # Guarda cache si se puede
    try:
        df.to_parquet(cache_file, index=False)
    except Exception:
        pass

    return df


def load_demand(system: str, start_dt: datetime, end_dt: datetime, batch_days: int = 7) -> pd.DataFrame:
    """Une batches y regresa una serie completa (timestamp, demand_mw)."""
    parts = []

    for a, b in _date_range_batches(start_dt, end_dt, batch_days=batch_days):
        df_part = fetch_cenace(system, a, b)
        if isinstance(df_part, pd.DataFrame) and not df_part.empty:
            parts.append(df_part)

    if not parts:
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    df = pd.concat(parts, ignore_index=True)

    df = (
        df.drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    return df


def quality_report(df: pd.DataFrame):
    """Reporte simple de calidad + chequeos básicos."""
    if df.empty:
        return {
            "rows": 0,
            "nan_values": None,
            "negatives": None,
            "duplicates": None,
            "time_gaps": None,
        }

    rows = int(len(df))
    nan_values = int(df["demand_mw"].isna().sum())
    negatives = int((df["demand_mw"] < 0).sum())
    duplicates = int(df["timestamp"].duplicated().sum())

    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=65)]
    time_gaps = int(len(gaps))

    return {
        "rows": rows,
        "nan_values": nan_values,
        "negatives": negatives,
        "duplicates": duplicates,
        "time_gaps": time_gaps,
    }

# ----------------------------
# UI
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    system = st.selectbox("Sistema", ["SIN", "BCA", "BCS"], index=0)
with col2:
    days = st.slider("Días a descargar (para probar)", min_value=1, max_value=14, value=7)

end_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
start_dt = end_dt - timedelta(days=int(days))

st.caption(f"Rango: {start_dt} → {end_dt} (batching de {min(7, int(days))} días)")

if st.button("Descargar y graficar"):
    with st.spinner("Descargando desde CENACE (con cache + batching)..."):
        df = load_demand(system=system, start_dt=start_dt, end_dt=end_dt, batch_days=min(7, int(days)))

    st.subheader("Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Gráfica")
    if not df.empty:
        st.line_chart(df.set_index("timestamp")["demand_mw"])
    else:
        st.warning("No llegaron datos (df vacío).")

    st.subheader("Reporte de calidad")
    st.json(quality_report(df))
