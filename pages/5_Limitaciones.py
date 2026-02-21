import streamlit as st

st.title("Limitaciones del Modelo")

st.markdown("""
Este simulador:

- No modela red de transmisión interna (modelo agregado por sistema)
- No incluye unit commitment
- No considera pérdidas eléctricas
- Es una aproximación académica simplificada
""")
