#src/model_deploy.py

#liberias
import pandas as pd
import numpy as np
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

#1. Inicializacion de la aplicación FastAPI
app = FastAPI(
    title = "API de Predicción de Pago a Tiempo",
    description = "Esta API permite predecir si un cliente pagará a tiempo o no, utilizando un modelo",
    version = "1.1.1"
)

#2. Acá vamos a cargar el modelo (modelo.pkl)
try:
    #2.1 Aca cargamos el modelo desde el archivo .pkl
    with open("../models/modelo.pkl", "rb") as f:
        modelo = pickle.load(f)

    print("Modelo cargado exitosamente...")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

# 3. Definición de endpoints
@app.get("/saludo")
def saludo():
    return {"mensaje": "Hola! Esta es una API para predecir si un cliente pagará a tiempo o no"}

# 4. Ahora vamos a definir un endpoint para hacer predicciones
@app.post("/predict")
def predict_batch (input_data: dict):
    if modelo is None: 
        return {"El modelo no pudo ser cargado. Revisa los logs del servidor para mas detalles"}
    
    try:
        return {"El modelo está cargado y listo para hacer predicciones"}

    except Exception as e:
        return {f"Error al hacer la predicción: {e}"}
    
    modelo.predict