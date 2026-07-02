# src/model_monitoring.py
# Deteccion de Data Drift con PSI (numericas) y Chi-Cuadrado (categoricas)
#
# QUE HACE ESTE ARCHIVO:
#   Compara los datos ORIGINALES (con los que se entreno el modelo)
#   contra los datos NUEVOS (con drift simulado) para ver si las
#   variables cambiaron su distribucion.
#
#   Si cambiaron mucho -> el modelo puede fallar -> hay que reentrenar.
#

import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from model_utils import preparar_datos
from cargar_datos import cargarDatos


def calcular_psi(original: np.ndarray, nuevo: np.ndarray, bins: int = 10) -> float:
    """
    Mide cuanto cambio una variable NUMERICA entre dos grupos (PSI).

    PSI = Population Stability Index.

    Como funciona (simple):
      1. Divide los valores en 10 "cajones" segun los datos originales
      2. Cuenta cuantos caen en cada cajon en ORIGINAL vs NUEVO
      3. Si las proporciones cambiaron mucho -> PSI alto -> hay drift

    Umbrales:
      PSI < 0.1  = SIN CAMBIO  (verde)
      0.1 - 0.2  = MODERADO    (amarillo)
      PSI > 0.2  = DRIFT       (rojo)

    Parametros:
        original: np.ndarray - Datos de referencia (los del entrenamiento)
        nuevo: np.ndarray    - Datos nuevos a comparar
        bins: int            - Cantidad de cajones (default 10)

    Retorna:
        float - El valor del PSI
    """
    # Paso 1: crear los bordes de los 10 cajones basados en los datos originales
    # Ej: si las edades van de 18 a 70, los bordes serian [18, 23, 28, ..., 70]
    _, bordes = np.histogram(original, bins=bins)

    # Paso 2: contar cuantos datos originales caen en cada cajon
    # y dividir por el total para obtener PROPORCIONES
    orig_binned = np.histogram(original, bins=bordes)[0] / len(original)

    # Paso 3: hacer lo mismo con los datos nuevos (usando los MISMOS bordes)
    nuevo_binned = np.histogram(nuevo, bins=bordes)[0] / len(nuevo)

    # Paso 4: evitar division por cero.
    # Si un cajon quedo vacio (proporcion = 0), le ponemos 0.0001
    orig_binned = np.where(orig_binned == 0, 0.0001, orig_binned)
    nuevo_binned = np.where(nuevo_binned == 0, 0.0001, nuevo_binned)

    # Paso 5: formula del PSI
    # Suma de (nuevo - original) * log(nuevo / original) para cada cajon
    psi = np.sum((nuevo_binned - orig_binned) * np.log(nuevo_binned / orig_binned))

    return psi


def calcular_chi2(original: pd.Series, nuevo: pd.Series) -> tuple:
    """
    Mide cuanto cambio una variable CATEGORICA entre dos grupos.

    Sirve para variables de texto (ej: tipo_laboral = "empleado", "independiente").
    Usa la prueba estadistica Chi-Cuadrado.

    Parametros:
        original: pd.Series - Datos de referencia
        nuevo: pd.Series    - Datos nuevos a comparar

    Retorna:
        tuple: (chi2, p_valor)
            chi2    - Que tan diferentes son (numero grande = mas diferente)
            p_valor - Probabilidad de que la diferencia sea casual.
                      Si p_valor < 0.05 -> hay DRIFT (diferencia real, no casual)
    """
    # Cruza las categorias de ambos grupos en una tabla
    # Ej: tipo_laboral en ORIGINAL vs NUEVO
    tabla = pd.crosstab(original.fillna("DESCONOCIDO"), nuevo.fillna("DESCONOCIDO"), margins=False)
    # Aplica la prueba estadistica Chi-Cuadrado
    chi2, p_valor, _, _ = chi2_contingency(tabla)
    return chi2, p_valor


def asignar_alerta_psi(psi: float) -> str:
    """
    Convierte un valor de PSI en un semaforo (alerta).

    Parametros:
        psi: float - Valor del PSI calculado

    Retorna:
        str - "SIN CAMBIO" si PSI < 0.1
              "MODERADO"   si 0.1 <= PSI < 0.2
              "DRIFT"      si PSI >= 0.2
    """
    if psi < 0.1:
        return "SIN CAMBIO"    # Verde
    elif psi < 0.2:
        return "MODERADO"      # Amarillo
    return "DRIFT"             # Rojo


def recomendar(df_resultados: pd.DataFrame) -> list:
    """
    Genera mensajes segun cuantas variables tienen drift.

    Mira el porcentaje de features en DRIFT y MODERADO y da
    recomendaciones: reentrenar urgente, monitorear, o que esta ok.

    Parametros:
        df_resultados: pd.DataFrame - Tabla con columna "Alerta"

    Retorna:
        list[str] - Lista de mensajes de recomendacion
    """
    recomendaciones = []
    drift_count = len(df_resultados[df_resultados["Alerta"] == "DRIFT"])
    moderado_count = len(df_resultados[df_resultados["Alerta"] == "MODERADO"])
    total = len(df_resultados)

    # Si mas del 30% de las variables tienen drift -> critico
    if drift_count > total * 0.3:
        recomendaciones.append("CRITICO: Mas del 30% de las features tienen DRIFT. Se recomienda reentrenar el modelo urgentemente.")
    # Si entre 15-30% -> alerta
    elif drift_count > total * 0.15:
        recomendaciones.append("ALERTA: Entre 15-30% de features con DRIFT. Monitorear de cerca y preparar reentrenamiento.")
    # Si al menos 1 -> atencion
    elif drift_count > 0:
        recomendaciones.append(f"ATENCION: {drift_count} feature(s) con DRIFT. Revisar las variables afectadas.")

    if moderado_count > 0:
        recomendaciones.append(f"REVISION: {moderado_count} feature(s) con cambios moderados. Programar revision para proximo ciclo.")

    # Si todo esta bien
    if drift_count == 0 and moderado_count == 0:
        recomendaciones.append("OK: Ninguna feature presenta cambios significativos. El modelo sigue siendo valido.")

    return recomendaciones


def analisis_temporal() -> pd.DataFrame:
    """
    Analiza como cambio el drift MES A MES dentro del dataset original.

    Agrupa los datos por fecha_prestamo y compara la distribucion de
    cada mes contra el dataset COMPLETO.

    IMPORTANTE: Esto usa el dataset ORIGINAL (Base_de_datos.xlsx), NO el
    de drift. Los meses con pocos registros (principio y final del rango)
    tienden a dar PSI alto artificialmente.

    Retorna:
        pd.DataFrame - Tabla con columnas: Periodo, Muestras, PSI_promedio
    """
    print("\n" + "=" * 85)
    print(" ANALISIS TEMPORAL - Evolucion del Drift")
    print("=" * 85)

    # Cargar el dataset ORIGINAL (NO el de drift)
    ruta_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Base_de_datos.xlsx"
    )
    df_raw = pd.read_excel(ruta_base)
    fechas = df_raw["fecha_prestamo"].copy()
    df = preparar_datos(df_raw)

    # Convertir fecha a formato mes (ej: "2025-01")
    fechas = pd.to_datetime(fechas, errors="coerce")
    df["periodo"] = fechas.dt.to_period("M").astype(str)

    target = "Pago_atiempo"

    # Cargar el preprocessor (el mismo que se uso al entrenar)
    ruta_preprocessor = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "preprocessor.pkl"
    )
    preprocessor = joblib.load(ruta_preprocessor)

    X_all = df.drop(columns=[target])
    X_all_t = preprocessor.transform(X_all)

    periodos = sorted(df["periodo"].unique())

    print(f"\nPeriodos encontrados: {len(periodos)}")
    print(f"Rango: {periodos[0]} a {periodos[-1]}")

    resultados_temp = []
    for periodo in periodos:
        idx = df["periodo"] == periodo
        X_periodo = X_all_t[idx]

        # Si el mes tiene menos de 50 registros, lo salteamos
        # (pocos datos = PSI poco confiable)
        if X_periodo.shape[0] < 50:
            continue

        # Calcular PSI entre ESTE mes y el dataset COMPLETO
        psi_total = 0
        count = 0
        for i in range(X_all_t.shape[1]):
            col_orig = X_all_t[:, i]      # TODOS los datos
            col_periodo = X_periodo[:, i]  # Solo los de este mes
            # Solo calculamos PSI si la variable tiene mas de 2 valores distintos
            if len(np.unique(col_orig)) > 2:
                psi_total += calcular_psi(col_orig, col_periodo)
                count += 1

        # PSI promedio de todas las variables para este mes
        psi_prom = psi_total / count if count else 0
        size = X_periodo.shape[0]

        resultados_temp.append({
            "Periodo": periodo,
            "Muestras": size,
            "PSI_promedio": round(psi_prom, 4)
        })

    df_temp = pd.DataFrame(resultados_temp)

    print("\n Evolucion temporal del drift:")
    print(df_temp.to_string(index=False))

    return df_temp


def detectar_drift():
    """
    Funcion PRINCIPAL del archivo. Orquesta la deteccion de Data Drift.

    Proceso:
      1. Carga los datos ORIGINALES (con los que se entreno el modelo)
      2. Carga los datos NUEVOS (archivo con drift simulado)
      3. Variables CATEGORICAS -> aplica Chi-Cuadrado (p < 0.05 = DRIFT)
      4. Variables NUMERICAS   -> aplica PSI (> 0.2 = DRIFT)
      5. Genera tabla con resultados por variable
      6. Da recomendaciones automaticas

    Retorna:
        pd.DataFrame - Tabla con columnas: Feature, Metrica, Valor, p_valor, Alerta
    """
    print("\n" + "=" * 85)
    print(" DETECCION DE DATA DRIFT")
    print(" Metricas: PSI (numericas) | Chi-Cuadrado (categoricas)")
    print("=" * 85)

    # ================================================================
    # PASO 1: Cargar dataset ORIGINAL
    # ================================================================
    print("\n[1] Cargando dataset original...")
    df_original = cargarDatos()        # Carga Base_de_datos.xlsx
    df_original = preparar_datos(df_original)  # Limpia y crea nuevas columnas

    # ================================================================
    # PASO 2: Cargar dataset con DRIFT SIMULADO
    # ================================================================
    ruta_drift = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Base_de_datos_con_Data_Drift_Simulado.xlsx"
    )
    print(f"[2] Cargando dataset con drift desde: {ruta_drift}")
    df_nuevo = pd.read_excel(ruta_drift)
    df_nuevo = preparar_datos(df_nuevo)

    # ================================================================
    # PASO 3: Cargar el preprocessor (para transformar los datos
    #          igual que cuando se entreno el modelo)
    # ================================================================
    ruta_preprocessor = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "preprocessor.pkl"
    )
    print(f"[3] Cargando preprocessor desde: {ruta_preprocessor}")
    preprocessor = joblib.load(ruta_preprocessor)

    target = "Pago_atiempo"
    X_orig = df_original.drop(columns=[target])  # Solo las features (sin la columna a predecir)
    X_nuevo = df_nuevo.drop(columns=[target])

    # ================================================================
    # PASO 4: Identificar que tipo es cada variable
    # ================================================================
    # - Numericas: edad, salario, etc.
    # - Categoricas: tipo_laboral, etc. (texto)
    # - Ordinales: tendencia_ingresos (Decreciente < Estable < Creciente)
    num_features = X_orig.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = [
        c for c in X_orig.select_dtypes(include=["object", "category"]).columns
        if c != "tendencia_ingresos"
    ]
    ord_features = ["tendencia_ingresos"]

    # ================================================================
    # PASO 5: Chi-Cuadrado para variables CATEGORICAS
    # ================================================================
    print(f"\n[4] Chi-Cuadrado para {len(cat_features)} variables categoricas...")
    resultados = []
    for col in cat_features:
        chi2, p_valor = calcular_chi2(X_orig[col], X_nuevo[col])
        # Si p_valor < 0.05 -> la diferencia es estadisticamente significativa -> DRIFT
        alerta = "DRIFT" if p_valor < 0.05 else "SIN CAMBIO"
        resultados.append({
            "Feature": col,
            "Metrica": "Chi2",          # Que metrica se uso
            "Valor": round(chi2, 2),    # El valor de Chi-Cuadrado
            "p_valor": round(p_valor, 4), # La probabilidad de que sea casual
            "Alerta": alerta
        })

    # ================================================================
    # PASO 6: PSI para variables NUMERICAS
    # ================================================================
    # Transformamos los datos con el preprocessor (imputar nulos,
    # escalar, one-hot encoding... igual que en entrenamiento)
    X_orig_t = preprocessor.transform(X_orig)
    X_nuevo_t = preprocessor.transform(X_nuevo)

    # Intentar recuperar los nombres de las columnas despues de la
    # transformacion (one-hot encoding crea muchas columnas nuevas)
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        ohe_nombres = ohe.get_feature_names_out(cat_features).tolist()
    except Exception:
        ohe_nombres = []
    nombres_post = num_features + ohe_nombres + ord_features

    # Si por alguna razon no coinciden las cantidades, ponemos nombres genericos
    if len(nombres_post) != X_orig_t.shape[1]:
        nombres_post = [f"feature_{i}" for i in range(X_orig_t.shape[1])]

    print(f"[5] PSI para {len(nombres_post)} features numericas/ordinales...")
    for i, nombre in enumerate(nombres_post):
        col_orig = X_orig_t[:, i]    # Columna i del dataset original (transformado)
        col_nuevo = X_nuevo_t[:, i]  # Columna i del dataset nuevo (transformado)

        # Solo calculamos PSI si la variable tiene mas de 2 valores distintos
        # (variables binarias como 0/1 no tienen sentido con PSI)
        if len(np.unique(col_orig)) > 2:
            psi = round(calcular_psi(col_orig, col_nuevo), 4)
            alerta = asignar_alerta_psi(psi)
            resultados.append({
                "Feature": nombre,
                "Metrica": "PSI",       # Que metrica se uso
                "Valor": psi,           # El valor del PSI
                "p_valor": "-",         # No aplica para PSI
                "Alerta": alerta
            })

    # Convertir la lista de resultados en una tabla (DataFrame)
    # y ordenarla: primero las que tienen DRIFT, luego MODERADO, luego SIN CAMBIO
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values("Alerta", ascending=False).reset_index(drop=True)

    # ================================================================
    # PASO 7: Mostrar reporte final
    # ================================================================
    print("\n" + "=" * 85)
    print(" REPORTE DE DATA DRIFT")
    print("=" * 85)
    print(df_resultados.to_string(index=False))

    print("\n RESUMEN:")
    print(f"  Features con DRIFT:     {len(df_resultados[df_resultados['Alerta'] == 'DRIFT'])}")
    print(f"  Features con MODERADO:  {len(df_resultados[df_resultados['Alerta'] == 'MODERADO'])}")
    print(f"  Features SIN CAMBIO:    {len(df_resultados[df_resultados['Alerta'] == 'SIN CAMBIO'])}")

    # ================================================================
    # PASO 8: Recomendaciones
    # ================================================================
    print("\n" + "=" * 85)
    print(" RECOMENDACIONES")
    print("=" * 85)
    for rec in recomendar(df_resultados):
        print(f"  >> {rec}")

    return df_resultados


# -------------------------------------------------------------------
# Si ejecutan este archivo directamente (python model_monitoring.py)
# corre la deteccion de drift completa
# -------------------------------------------------------------------
if __name__ == "__main__":
    detectar_drift()
