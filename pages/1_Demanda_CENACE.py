import os
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

BASE_WS = "https://ws01.cenace.gob.mx:8082/SWPDEZC/SIM"
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

st.title("Demanda CENACE (Semana 2)")
st.write("Descarga por API oficial de CENACE + batching (≤7 días) + cache en disco + checks de calidad.")


def _date_range_batches(start_dt, end_dt, batch_days=7):
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=batch_days), end_dt)
        yield cur, nxt
        cur = nxt


def _cache_path(system, start_dt, end_dt):
    s = start_dt.strftime("%Y%m%d")
    e = end_dt.strftime("%Y%m%d")
    return os.path.join(CACHE_DIR, f"demanda_{system}_{s}_{e}.parquet")


def _parse_cenace_response(data, system):
    resultados = None

    if "Resultados" in data:
        resultados = data["Resultados"]
    elif isinstance(data, list):
        resultados = data
    else:
        for v in data.values():
            if isinstance(v, list) and len(v) > 0:
                resultados = v
                break

    if not resultados:
        raise ValueError(f"No encontré datos. Claves: {list(data.keys()) if isinstance(data, dict) else type(data)}")

    rows = []
    for r in resultados:
        if not isinstance(r, dict):
            continue
        fecha = r.get("fecha") or r.get("Fecha") or r.get("FECHA", "")
        hora_raw = r.get("hora") or r.get("Hora") or r.get("HORA", "1")
        demanda_raw = (r.get("Demanda") or r.get("demanda") or
                       r.get("valor") or r.get("Valor") or
                       r.get("pronostico") or r.get("Pronostico"))
        try:
            hora = int(float(hora_raw)) - 1
            ts = datetime.strptime(fecha, "%Y-%m-%d") + timedelta(hours=hora)
            demand_val = float(demanda_raw)
            rows.append({"timestamp": ts, "demand_mw": demand_val})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_cenace(system, start_dt, end_dt):
    cache_file = _cache_path(system, start_dt, end_dt)
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    url = (
        f"{BASE_WS}/{system}/MDA/{system}"
        f"/{start_dt.strftime('%Y')}/{start_dt.strftime('%m')}/{start_dt.strftime('%d')}"
        f"/{end_dt.strftime('%Y')}/{end_dt.strftime('%m')}/{end_dt.strftime('%d')}"
        f"/JSON"
    )

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    try:
        r = requests.get(url, headers=headers, timeout=60, verify=True)
    except requests.exceptions.SSLError:
        r = requests.get(url, headers=headers, timeout=60, verify=False)

    if r.status_code != 200:
        st.error(f"CENACE respondió {r.status_code} para {system}")
        st.error(f"URL consultada: {url}")
        st.error(r.text[:300])
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    try:
        data = r.json()
    except Exception as e:
        st.error(f"No se pudo leer el JSON: {e}")
        st.error(r.text[:300])
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    try:
        df = _parse_cenace_response(data, system)
    except Exception as e:
        st.error("No pude parsear la respuesta.")
        st.error(str(e))
        st.json(data if isinstance(data, dict) else {})
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].copy()

    try:
        df.to_parquet(cache_file, index=False)
    except Exception:
        pass

    return df


def load_demand(system, start_dt, end_dt, batch_days=7):
    parts = []
    for a, b in _date_range_batches(start_dt, end_dt, batch_days=batch_days):
        df_part = fetch_cenace(system, a, b)
        if isinstance(df_part, pd.DataFrame) and not df_part.empty:
            parts.append(df_part)

    if not parts:
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def quality_report(df):
    if df.empty:
        return {"rows": 0, "nan_values": None, "negatives": None, "duplicates": None, "time_gaps": None}

    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=65)]

    return {
        "rows": int(len(df)),
        "nan_values": int(df["demand_mw"].isna().sum()),
        "negatives": int((df["demand_mw"] < 0).sum()),
        "duplicates": int(df["timestamp"].duplicated().sum()),
        "time_gaps": int(len(gaps)),
    }


# ----------------------------
# UI
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    system = st.selectbox("Sistema", ["SIN", "BCA", "BCS"], index=0)
with col2:
    days = st.slider("Días a descargar", min_value=1, max_value=14, value=7)

end_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=8)
start_dt = end_dt - timedelta(days=int(days))

st.caption(f"Rango: {start_dt.date()} → {end_dt.date()}")

with st.expander("🔧 Ver URL que se consultará"):
    url_preview = (
        f"{BASE_WS}/{system}/MDA/{system}"
        f"/{start_dt.strftime('%Y')}/{start_dt.strftime('%m')}/{start_dt.strftime('%d')}"
        f"/{end_dt.strftime('%Y')}/{end_dt.strftime('%m')}/{end_dt.strftime('%d')}"
        f"/JSON"
    )
    st.code(url_preview)

if st.button("Descargar y graficar"):
    with st.spinner("Descargando desde CENACE..."):
        df = load_demand(system=system, start_dt=start_dt, end_dt=end_dt, batch_days=min(7, int(days)))

    st.subheader("Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Gráfica")
    if not df.empty:
        st.line_chart(df.set_index("timestamp")["demand_mw"])
    else:
        st.warning("No llegaron datos.")

    st.subheader("Reporte de calidad")
    st.json(quality_report(df))
