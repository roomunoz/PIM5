# ARCHIVO: src/ft_engineering.py
# VERSION: V1.1.0 (Feature Engineering + Modelado Supervisado)
# Descripcion: Pipeline de feature engineering con 3 modelos base.
# Incluye funciones summarize_classification y build_model.

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from model_utils import preparar_datos
from model_training_evaluation import CONFIG_MODELOS, entrenar_y_evaluar, comparar_modelos

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from cargar_datos import cargarDatos


def construir_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    ordinal_features = ['tendencia_ingresos']
    cat_features = [
        c for c in X.select_dtypes(include=['object', 'category', 'str']).columns
        if c not in ordinal_features
    ]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            categories=[['Decreciente', 'Estable', 'Creciente']]
        ))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features),
        ('ord', ord_pipe, ordinal_features)
    ])

    return preprocessor


def ft_engineering():
    print("\n" + "="*85)
    print(" PIPELINE FEATURE ENGINEERING + MODELADO SUPERVISADO")
    print("="*85)

    df = cargarDatos()
    print(f"\n[LOG] Dataset cargado — Shape inicial: {df.shape}")
    df = preparar_datos(df)

    target = 'Pago_atiempo'
    X = df.drop(columns=[target])
    y = df[target]

    print(f"[LOG] Features finales: {X.shape}")
    print(f"[LOG] Distribucion target:\n{y.value_counts(normalize=True)}")

    preprocessor = construir_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print(f"[LOG] Train shape: {X_train_t.shape}")
    print(f"[LOG] Test shape:  {X_test_t.shape}")

    print("\n" + "="*85)
    print(" ENTRENAMIENTO DE MODELOS")
    print("="*85)

    modelos = {}
    metricas_lista = []
    for nombre, model_class, kwargs in CONFIG_MODELOS:
        print(f"\n[LOG] Entrenando {nombre}...")
        model, metrica = entrenar_y_evaluar(
            model_class, kwargs.copy(), nombre,
            X_train_t, y_train, X_test_t, y_test
        )
        modelos[nombre.lower().replace(' ', '_')] = model
        metricas_lista.append(metrica)

    df_comparativa = comparar_modelos(metricas_lista)

    mejor_nombre = df_comparativa.iloc[0]['Modelo']
    mejor_modelo = modelos[mejor_nombre.lower().replace(' ', '_')]
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    ruta_modelo = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo.pkl')
    ruta_preprocessor = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl')

    joblib.dump(mejor_modelo, ruta_modelo)
    joblib.dump(preprocessor, ruta_preprocessor)
    print(f"\n[LOG] Mejor modelo guardado: {ruta_modelo}")
    print(f"[LOG] Preprocessor guardado: {ruta_preprocessor}")

    print("\n PIPELINE FINALIZADO CORRECTAMENTE")

    return (
        X_train_t, X_test_t, y_train, y_test,
        modelos,
        df_comparativa
    )


if __name__ == "__main__":
    try:
        print("\n" + "="*85)
        print(" EJECUCION LOCAL DEL PIPELINE")
        print("="*85)
        X_train_p, X_test_p, y_train_p, y_test_p, modelos, comparativa = ft_engineering()
        print("\n PIPELINE EJECUTADO CORRECTAMENTE")
    except Exception as e:
        print("\n ERROR EN PIPELINE")
        print(f"Detalle: {e}")
        raise