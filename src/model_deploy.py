# src/model_deploy.py
# API de prediccion de riesgo crediticio (Pago a tiempo) con FastAPI

import os
import sys
from typing import List, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import preparar_datos

# =====================================================
# RUTAS DEL PROYECTO
# =====================================================
RUTA_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(RUTA_PROYECTO, "models", "modelo.pkl")
RUTA_PREPROC = os.path.join(RUTA_PROYECTO, "models", "preprocessor.pkl")

# =====================================================
# INICIALIZACION DE LA APP
# =====================================================
app = FastAPI(
    title="API de Prediccion de Pago a Tiempo",
    description="Predice si un cliente pagara a tiempo o no, en base al modelo "
                 "de riesgo crediticio entrenado con datos historicos.",
    version="1.1.1"
)

# Carga del modelo y el preprocessor al iniciar la app
try:
    modelo = joblib.load(RUTA_MODELO)
    preprocessor = joblib.load(RUTA_PREPROC)
    print("[LOG] Modelo y preprocessor cargados exitosamente")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo o el preprocessor: {e}")
    modelo = None
    preprocessor = None


# =====================================================
# ESQUEMAS DE ENTRADA (PYDANTIC)
# =====================================================
class Cliente(BaseModel):
    """
    Representa un cliente sobre el que se quiere predecir.
    Debe incluir todas las columnas del dataset original (Base_de_datos.xlsx)
    excepto la target (Pago_atiempo).
    """

    # Identificacion (opcionales, se descartan internamente)
    id_cliente: Optional[str] = None
    nombre_cliente: Optional[str] = None

    # Datos del prestamo
    tipo_credito: int
    fecha_prestamo: str
    capital_prestado: float
    plazo_meses: int

    # Datos del cliente
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float

    # Historial crediticio
    puntaje: Optional[float] = None
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str


class PrediccionRequest(BaseModel):
    clientes: List[Cliente]


class PrediccionResponse(BaseModel):
    prediccion: int
    probabilidad_mora: float


# =====================================================
# ENDPOINTS
# =====================================================
@app.get("/saludo")
def saludo():
    return {"mensaje": "API para predecir si un cliente pagara a tiempo o no"}


@app.get("/health")
def health():
    """Chequeo simple de que el modelo y el preprocessor estan cargados."""
    return {
        "modelo_cargado": modelo is not None,
        "preprocessor_cargado": preprocessor is not None
    }


@app.post("/predict", response_model=List[PrediccionResponse])
def predict(request: PrediccionRequest):
    if modelo is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo o el preprocessor no se cargaron correctamente. "
                   "Revisar que existan models/modelo.pkl y models/preprocessor.pkl"
        )

    if not request.clientes:
        raise HTTPException(status_code=400, detail="La lista 'clientes' esta vacia")

    try:
        # Convertimos los clientes (Pydantic) a DataFrame
        df = pd.DataFrame([c.model_dump() for c in request.clientes])

        # Mismo pipeline de limpieza/features que en entrenamiento
        df = preparar_datos(df)

        # Nos aseguramos de no pasarle la columna target si vino por error
        X = df.drop(columns=["Pago_atiempo"], errors="ignore")

        # Transformacion con el MISMO preprocessor usado en entrenamiento
        X_t = preprocessor.transform(X)

        preds = modelo.predict(X_t)
        probas = modelo.predict_proba(X_t)[:, 1]

        resultados = [
            {"prediccion": int(pred), "probabilidad_mora": round(float(prob), 4)}
            for pred, prob in zip(preds, probas)
        ]

        return resultados

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Falta una columna requerida en los datos del cliente: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
