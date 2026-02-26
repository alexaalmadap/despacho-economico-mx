import os
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Config
# ----------------------------
# ✅ CAMBIO 1: Usamos la API oficial de CENACE (ws01), no la página web
BASE_WS = "https://ws01.cenace.gob.mx:8082/SWEDREZC/SIM"

# Las zonas de carga que representan cada sistema completo
# ✅ CAMBIO 2: Cada sistema tiene su zona de carga principal
ZONAS = {
    "SIN": "SIN",    # Sistema Interconectado Nacional
    "BCA": "BCA",    # Baja California
    "BCS": "BCS",    # Baja California Sur
}

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

st.title("Demanda CENACE (Semana 2)")
st.write("Descarga por API oficial de CENACE + batching (≤7 días) + cache en disco + checks de calidad.")

# ----------------------------
# Helpers
# ----------------------------

def _date_range_batches(start_dt: datetime, end_dt: datetime, batch_days: int = 7):
    """Genera rangos en bloques de batch_days días."""
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


def _parse_cenace_response(data: dict, system: str) -> pd.DataFrame:
    """
    Convierte la respuesta de la API oficial ws01 a DataFrame con columnas:
    timestamp, demand_mw
    
    La respuesta tiene esta forma:
    {
      "Resultados": [
        {
          "zona_carga": "SIN",
          "fecha": "2025-01-01",
          "hora": "1",
          "demanda": "35000.5"
        }, ...
      ]
    }
    """
    # ✅ CAMBIO 3: Parseamos la estructura real de la API oficial
    if "Resultados" not in data:
        raise ValueError(f"La respuesta no tiene 'Resultados'. Claves recibidas: {list(data.keys())}")
    
    resultados = data["Resultados"]
    
    if not resultados:
        return pd.DataFrame(columns=["timestamp", "demand_mw"])
    
    rows = []
    for r in resultados:
        # La API devuelve fecha y hora por separado
        # Hora va de 1 a 24, donde hora 1 = 00:00, hora 24 = 23:00
        fecha = r.get("fecha", "")
        hora = int(r.get("hora", 1)) - 1  # convertir a 0-23
        demanda = r.get("demanda", None)
        
        try:
            ts = datetime.strptime(fecha, "%Y-%m-%d") + timedelta(hours=hora)
            demand_val = float(demanda)
            rows.append({"timestamp": ts, "demand_mw": demand_val})
        except Exception:
            continue  # Si algún registro falla, lo saltamos
    
    if not rows:
        return pd.DataFrame(columns=["timestamp", "demand_mw"])
    
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_cenace(system: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Descarga demanda de CENACE usando la API oficial REST (GET).
    Devuelve DataFrame con columnas: timestamp, demand_mw
    """
    cache_file = _cache_path(system, start_dt, end_dt)
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass  # Si falla el cache, continuamos y re-descargamos

    zona = ZONAS.get(system, system)
    
    # ✅ CAMBIO 4: Construimos la URL con los parámetros en la ruta (método GET, no POST)
    url = (
        f"{BASE_WS}/{system}/{zona}"
        f"/{start_dt.strftime('%Y')}/{start_dt.strftime('%m')}/{start_dt.strftime('%d')}"
        f"/{end_dt.strftime('%Y')}/{end_dt.strftime('%m')}/{end_dt.strftime('%d')}"
        f"/JSON"
    )
    
    # ✅ CAMBIO 5: Usamos GET con headers simples, no POST con JSON body
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=60, verify=True)
    except requests.exceptions.SSLError:
        # ✅ CAMBIO 6: Fallback si hay problema de SSL (común en servidores gov)
        r = requests.get(url, headers=headers, timeout=60, verify=False)
    
    if r.status_code != 200:
        st.error(f"CENACE respondió {r.status_code} para {system}")
        st.error(f"URL consultada: {url}")
        st.error(r.text[:300])
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    try:
        data = r.json()
    except Exception as e:
        st.error(f"No se pudo leer el JSON de CENACE: {e}")
        st.error(r.text[:300])
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    try:
        df = _parse_cenace_response(data, system)
    except Exception as e:
        st.error("No pude parsear la respuesta de CENACE.")
        st.error(str(e))
        st.json(data if isinstance(data, dict) else {})
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    # Filtra exactamente al rango pedido
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].copy()

    # Guarda en cache
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
    """Reporte de calidad del dataset."""
    if df.empty:
        return {"rows": 0, "nan_values": None, "negatives": None, "duplicates": None, "time_gaps": None}

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

# ✅ CAMBIO 7: Usamos fechas pasadas (la API no tiene datos futuros ni de "hoy")
end_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=8)
start_dt = end_dt - timedelta(days=int(days))

st.caption(f"Rango: {start_dt.date()} → {end_dt.date()} (batching de {min(7, int(days))} días)")

# ✅ CAMBIO 8: Mostramos la URL que se va a consultar para debugging
with st.expander("🔧 Ver URL que se consultará"):
    zona = ZONAS.get(system, system)
    url_preview = (
        f"{BASE_WS}/{system}/{zona}"
        f"/{start_dt.strftime('%Y')}/{start_dt.strftime('%m')}/{start_dt.strftime('%d')}"
        f"/{end_dt.strftime('%Y')}/{end_dt.strftime('%m')}/{end_dt.strftime('%d')}"
        f"/JSON"
    )
    st.code(url_preview)

if st.button("Descargar y graficar"):
    with st.spinner("Descargando desde CENACE (con cache + batching)..."):
        df = load_demand(system=system, start_dt=start_dt, end_dt=end_dt, batch_days=min(7, int(days)))

    st.subheader("Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Gráfica")
    if not df.empty:
        st.line_chart(df.set_index("timestamp")["demand_mw"])
    else:
        st.warning("No llegaron datos. Revisa la URL en el expander de arriba.")

    st.subheader("Reporte de calidad")
    st.json(quality_report(df))
