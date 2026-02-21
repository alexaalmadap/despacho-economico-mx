import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st

# ----------------------------
# Config
# ----------------------------
CENACE_URL = "https://www.cenace.gob.mx/GraficaDemanda.aspx/obtieneValoresTotal"
CACHE_DIR = "data_cache"  # se crea automáticamente en Streamlit Cloud
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
    s = start_dt.strftime("%Y%m%d")
    e = end_dt.strftime("%Y%m%d")
    return os.path.join(CACHE_DIR, f"demanda_{system}_{s}_{e}.parquet")

def _parse_cenace_response(resp_json) -> pd.DataFrame:
    """
    Convierte la respuesta típica de servicios .aspx (a veces viene en resp_json['d'] como string JSON)
    a un DataFrame con columnas: timestamp, demand_mw
    """
    # Caso común en ASP.NET: {"d":"[...]"} donde d es string con JSON adentro
    if isinstance(resp_json, dict) and "d" in resp_json:
        payload = resp_json["d"]
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload
    else:
        data = resp_json

    # Intento de normalización: buscamos campos típicos (fecha/hora y valor)
    # Como CENACE puede cambiar nombres, hacemos heurística.
    df = pd.DataFrame(data)

    # Encuentra columnas candidatas
    col_time = None
    for c in df.columns:
        if str(c).lower() in ["fecha", "fechahora", "fecha_hora", "hora", "timestamp", "datetime"]:
            col_time = c
            break
    # A veces viene como "Fecha" o "fecha" o similar con texto
    if col_time is None:
        # toma la primera columna tipo string/datetime
        for c in df.columns:
            if df[c].dtype == object:
                col_time = c
                break

    col_val = None
    for c in df.columns:
        if str(c).lower() in ["valor", "mw", "demanda", "demanda_mw", "value", "y"]:
            col_val = c
            break
    if col_val is None:
        # toma la primera numérica
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                col_val = c
                break

    if col_time is None or col_val is None:
        raise ValueError(
            f"No pude identificar columnas de tiempo/valor. Columnas recibidas: {list(df.columns)}"
        )

    out = df[[col_time, col_val]].copy()
    out.columns = ["timestamp_raw", "demand_mw"]

    # Parse de timestamp: CENACE suele mandar strings
    out["timestamp"] = pd.to_datetime(out["timestamp_raw"], errors="coerce", dayfirst=True)
    out = out.drop(columns=["timestamp_raw"])

    # Limpieza valor
    out["demand_mw"] = pd.to_numeric(out["demand_mw"], errors="coerce")

    # Quita filas inválidas
    out = out.dropna(subset=["timestamp", "demand_mw"]).sort_values("timestamp").reset_index(drop=True)

    return out

def fetch_cenace(system: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Descarga demanda de CENACE para un sistema (SIN/BCA/BCS/Total según soporte del endpoint)
    por un rango. Usa cache en disco (parquet).
    """
    cache_file = _cache_path(system, start_dt, end_dt)
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # IMPORTANTE:
    # Este endpoint suele aceptar POST con JSON.
    # Como no tenemos el contrato exacto aquí, hacemos 2 intentos:
    # 1) Sin body (algunos devuelven lo más reciente)
    # 2) Con body típico (si falla el primero)
    headers = {"Content-Type": "application/json; charset=UTF-8"}

    # Intento 1: POST vacío
    try:
        r = requests.post(CENACE_URL, headers=headers, timeout=30)
        r.raise_for_status()
        df = _parse_cenace_response(r.json())
    except Exception:
        # Intento 2: body típico (puede que el endpoint requiera fechas y/o sistema)
        # OJO: esto puede necesitar ajuste según lo que realmente acepte CENACE.
        body = {
            "fechaInicio": start_dt.strftime("%d/%m/%Y"),
            "fechaFin": (end_dt - timedelta(days=1)).strftime("%d/%m/%Y"),
            "sistema": system
        }
        r = requests.post(CENACE_URL, headers=headers, data=json.dumps(body), timeout=30)
        r.raise_for_status()
        df = _parse_cenace_response(r.json())

    # Filtra rango por si el endpoint trae más datos de los pedidos
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].copy()

    # Guarda cache
    df.to_parquet(cache_file, index=False)
    return df

def load_demand(system: str, start_dt: datetime, end_dt: datetime, batch_days: int = 7) -> pd.DataFrame:
    """Une batches y regresa una serie completa."""
    parts = []
    for a, b in _date_range_batches(start_dt, end_dt, batch_days=batch_days):
        parts.append(fetch_cenace(system, a, b))
    if not parts:
        return pd.DataFrame(columns=["timestamp", "demand_mw"])
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)

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

    rows = len(df)
    nan_values = int(df["demand_mw"].isna().sum())
    negatives = int((df["demand_mw"] < 0).sum())
    duplicates = int(df["timestamp"].duplicated().sum())

    # Gaps: esperamos horario. Calcula diferencias
    diffs = df["timestamp"].diff().dropna()
    # gaps si diferencia > 1 hora (tolerancia 65 min)
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

# rango por defecto: últimos N días desde hoy (UTC local del server; suficiente para demo)
end_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
start_dt = end_dt - timedelta(days=int(days))

st.caption(f"Rango: {start_dt} → {end_dt} (batching de 7 días)")

if st.button("Descargar y graficar"):
    with st.spinner("Descargando desde CENACE (con cache + batching)..."):
        df = load_demand(system=system, start_dt=start_dt, end_dt=end_dt, batch_days=7)

    st.subheader("Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Gráfica")
    st.line_chart(df.set_index("timestamp")["demand_mw"])

    st.subheader("Reporte de calidad")
    rep = quality_report(df)
    st.json(rep)

    st.info("Si ves 0 filas o error de columnas, hay que ajustar el parse según el formato real que regresa CENACE.")
