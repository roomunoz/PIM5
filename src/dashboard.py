# src/dashboard.py
# Dashboard Streamlit - Monitoreo de Data Drift

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_monitoring import (
    detectar_drift, calcular_psi, calcular_chi2, recomendar, analisis_temporal
)
from model_utils import preparar_datos
from cargar_datos import cargarDatos
import joblib


st.set_page_config(
    page_title="Monitoreo de Modelo - Riesgo Crediticio",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def ejecutar_drift():
    return detectar_drift()


@st.cache_data
def cargar_datos_originales():
    df = cargarDatos()
    df = preparar_datos(df)
    return df


@st.cache_resource
def cargar_preprocessor():
    ruta = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "preprocessor.pkl"
    )
    return joblib.load(ruta)


st.title("📊 Monitoreo de Modelo - Riesgo Crediticio")

st.sidebar.header("Navegacion")
pagina = st.sidebar.radio(
    "Ir a:",
    ["Sobre el Proyecto", "Resumen de Drift", "Evolucion Temporal", "Detalle por Feature"]
)


# =====================================================
# PAGINA 1: SOBRE EL PROYECTO
# =====================================================
if pagina == "Sobre el Proyecto":
    st.header("Sobre el Proyecto")
    st.markdown("""
    ### Modelo de Riesgo Crediticio

    Proyecto Integrador - Data Science

    **Objetivo:** Desarrollar un modelo predictivo supervisado para anticipar el comportamiento de pago de nuevos solicitantes de credito.

    **Dataset:** ~10,700 registros historicos de creditos con 23 variables.

    **Modelos evaluados:** Logistic Regression, Random Forest, XGBoost.

    ### Data Drift

    El **Population Stability Index (PSI)** mide cambios en la distribucion de variables numericas entre el dataset original y nuevos datos.
    El **Chi-Cuadrado** hace lo mismo para variables categoricas.

    | Metrica | Tipo | Umbrales |
    |---|---|---|
    | **PSI** | Numericas | < 0.1 verde, 0.1-0.2 amarillo, > 0.2 rojo |
    | **Chi2** | Categoricas | p < 0.05 indica drift |
    """)


# =====================================================
# PAGINA 2: RESUMEN DE DRIFT
# =====================================================
elif pagina == "Resumen de Drift":
    st.header("Reporte de Data Drift")

    with st.spinner("Calculando drift..."):
        df_resultados = ejecutar_drift()

    col1, col2, col3 = st.columns(3)
    drift_count = len(df_resultados[df_resultados["Alerta"] == "DRIFT"])
    moderado_count = len(df_resultados[df_resultados["Alerta"] == "MODERADO"])
    seguro_count = len(df_resultados[df_resultados["Alerta"] == "SIN CAMBIO"])

    col1.metric("🔴 DRIFT", drift_count)
    col2.metric("🟡 MODERADO", moderado_count)
    col3.metric("🟢 SIN CAMBIO", seguro_count)

    # Recomendaciones integradas aqui
    st.subheader("Diagnostico")
    for rec in recomendar(df_resultados):
        if "CRITICO" in rec:
            st.error(f"🚨 {rec}")
        elif "OK" in rec:
            st.success(f"✅ {rec}")
        else:
            st.warning(f"⚠️ {rec}")

    # Semaforo
    total = len(df_resultados)
    if drift_count > total * 0.3:
        st.error("🔴 SEMAFORO: ROJO - Mas del 30% de las features tienen drift. Reentrenar urgente.")
    elif drift_count > total * 0.15:
        st.warning("🟡 SEMAFORO: AMARILLO - Entre 15-30% de features con drift. Preparar reentrenamiento.")
    elif drift_count > 0:
        st.info("🟠 SEMAFORO: NARANJA - Algunas features con drift. Monitorear.")
    else:
        st.success("🟢 SEMAFORO: VERDE - Modelo estable.")

    st.subheader("Tabla de Metricas")
    df_show = df_resultados.copy()

    def color_alerta(val):
        if val == "DRIFT":
            return "background-color: #ff4d4d; color: white"
        elif val == "MODERADO":
            return "background-color: #ffa726; color: white"
        return "background-color: #66bb6a; color: white"

    st.dataframe(
        df_show.style.map(color_alerta, subset=["Alerta"]),
        use_container_width=True,
        height=400
    )

    st.subheader("Graficos")
    col_graf1, col_graf2 = st.columns([1, 2])

    with col_graf1:
        counts = df_resultados["Alerta"].value_counts()
        fig, ax = plt.subplots(figsize=(3, 3))
        colors_map = {"DRIFT": "#ff4d4d", "MODERADO": "#ffa726", "SIN CAMBIO": "#66bb6a"}
        bars = ax.bar(counts.index, counts.values, color=[colors_map.get(c, "#ccc") for c in counts.index])
        ax.set_ylabel("Cantidad")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val), ha="center")
        st.pyplot(fig)

    with col_graf2:
        df_psi = df_resultados[df_resultados["Metrica"] == "PSI"].copy()
        if not df_psi.empty:
            df_psi["Valor"] = df_psi["Valor"].astype(float)
            df_psi = df_psi.sort_values("Valor", ascending=True)
            n_features = len(df_psi)
            alto = max(3, n_features * 0.35)
            fig2, ax2 = plt.subplots(figsize=(6, alto))
            bar_colors = ["#ff4d4d" if v > 0.2 else "#ffa726" if v > 0.1 else "#66bb6a" for v in df_psi["Valor"]]
            ax2.barh(df_psi["Feature"], df_psi["Valor"], color=bar_colors)
            ax2.axvline(0.1, color="#ffa726", linestyle="--", label="Mod (0.1)")
            ax2.axvline(0.2, color="#ff4d4d", linestyle="--", label="Drift (0.2)")
            ax2.set_xlabel("PSI")
            ax2.tick_params(axis="y", labelsize=7)
            ax2.legend(fontsize=7)
            fig2.tight_layout()
            st.pyplot(fig2)

    if drift_count > 0:
        st.info(f"🔴 {drift_count} feature(s) en DRIFT — Revisar estas variables para entender que cambio en el negocio.")
    if moderado_count > 0:
        st.info(f"🟡 {moderado_count} feature(s) con cambios MODERADOS — Programar revision para el proximo ciclo.")


# =====================================================
# PAGINA 3: EVOLUCION TEMPORAL
# =====================================================
elif pagina == "Evolucion Temporal":
    st.header("Evolucion del Drift a lo Largo del Tiempo")

    st.markdown("""
    Se agrupan los datos por **mes** usando `fecha_prestamo` y se calcula el PSI
    promedio de cada periodo contra la distribucion general del dataset completo.

    > ⚠️ **Limitacion**: Cada mes se compara contra el dataset **completo** (no contra
    > un periodo fijo de entrenamiento). Los meses con pocos registros (principio
    > o final del rango) suelen dar PSI alto aunque no haya drift real.
    > La deteccion de drift **real** esta en la pestania "Resumen de Drift".
    """)

    with st.spinner("Analizando evolucion temporal..."):
        df_temp = analisis_temporal()

    if not df_temp.empty:
        col1, col2 = st.columns(2)
        col1.metric("Periodos analizados", len(df_temp))
        col2.metric("Rango", f"{df_temp['Periodo'].iloc[0]} a {df_temp['Periodo'].iloc[-1]}")

        st.subheader("PSI promedio a lo largo del tiempo")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_temp["Periodo"], df_temp["PSI_promedio"], marker="o", color="#2196F3", linewidth=2)
        ax.axhline(0.1, color="#ffa726", linestyle="--", label="Moderado (0.1)")
        ax.axhline(0.2, color="#ff4d4d", linestyle="--", label="Drift (0.2)")
        ax.set_xlabel("Periodo")
        ax.set_ylabel("PSI Promedio")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Tabla de datos")
        st.dataframe(df_temp, use_container_width=True)

        st.subheader("Deteccion de Tendencias")
        ultimos = df_temp.tail(3)
        psi_tendencia = ultimos["PSI_promedio"].values

        # NOTA: Esta temporal compara cada mes contra el dataset COMPLETO
        # (incluyendo el mismo mes). Los meses con pocos datos (inicio y
        # final del periodo) tienden a tener PSI alto artificialmente.
        # Por eso se ve como una U: alto -> bajo -> alto.
        # NO es necesariamente drift progresivo.
        st.caption("(Cada mes se compara contra el dataset completo, no contra un periodo fijo de entrenamiento)")
        if len(psi_tendencia) >= 2 and psi_tendencia[-1] > psi_tendencia[0] * 1.5:
            st.warning("⚠️ Ultimos 3 periodos: el PSI subio. Revisar si es por muestras chicas o drift real.")
        elif len(psi_tendencia) >= 2 and psi_tendencia[-1] > 0.15:
            st.info("🔶 PSI elevado en el ultimo periodo. Monitorear.")
        else:
            st.success("✅ Sin tendencias preocupantes en los periodos recientes.")
    else:
        st.warning("No se pudieron generar periodos para el analisis temporal.")


# =====================================================
# PAGINA 4: DETALLE POR FEATURE
# =====================================================
elif pagina == "Detalle por Feature":
    st.header("Distribucion de Features Individuales")

    df_orig = cargar_datos_originales()
    ruta_drift = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Base_de_datos_con_Data_Drift_Simulado.xlsx"
    )
    df_nuevo_raw = pd.read_excel(ruta_drift)
    df_nuevo = preparar_datos(df_nuevo_raw)

    X_orig = df_orig.drop(columns=["Pago_atiempo"])
    X_nuevo = df_nuevo.drop(columns=["Pago_atiempo"])

    num_cols = X_orig.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_orig.select_dtypes(include=["object", "category"]).columns if c != "tendencia_ingresos"]
    ord_cols = ["tendencia_ingresos"]

    feature_elegida = st.selectbox("Seleccionar feature:", num_cols + ord_cols + cat_cols)

    col_graf, col_metricas = st.columns([3, 1])

    with col_graf:
        if feature_elegida in num_cols + ord_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(X_orig[feature_elegida].dropna(), bins=30, alpha=0.6, label="Original", color="#2196F3")
            ax.hist(X_nuevo[feature_elegida].dropna(), bins=30, alpha=0.6, label="Nuevo (con drift)", color="#FF5722")
            ax.set_xlabel(feature_elegida)
            ax.set_ylabel("Frecuencia")
            ax.set_title(f"Distribucion de {feature_elegida}")
            ax.legend()
            st.pyplot(fig)

        elif feature_elegida in cat_cols:
            df_comp = pd.DataFrame({
                "Original": X_orig[feature_elegida].fillna("DESCONOCIDO").value_counts(normalize=True),
                "Nuevo (con drift)": X_nuevo[feature_elegida].fillna("DESCONOCIDO").value_counts(normalize=True)
            }).fillna(0)
            fig, ax = plt.subplots(figsize=(10, 5))
            df_comp.plot(kind="bar", ax=ax, color=["#2196F3", "#FF5722"])
            ax.set_ylabel("Proporcion")
            ax.set_title(f"Distribucion de {feature_elegida}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    with col_metricas:
        st.subheader("Metricas")
        if feature_elegida in num_cols:
            psi = calcular_psi(
                X_orig[feature_elegida].fillna(X_orig[feature_elegida].median()).values,
                X_nuevo[feature_elegida].fillna(X_nuevo[feature_elegida].median()).values
            )
            st.metric("PSI", f"{psi:.4f}")

        elif feature_elegida in cat_cols:
            chi2, p_valor = calcular_chi2(X_orig[feature_elegida], X_nuevo[feature_elegida])
            st.metric("Chi2", f"{chi2:.2f}")
            st.metric("p-valor", f"{p_valor:.4f}")
