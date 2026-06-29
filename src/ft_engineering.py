# ARCHIVO: src/ft_engineering.py
# VERSION: V1.1.0 (Feature Engineering + Modelado Supervisado)
# Descripcion: Pipeline de feature engineering con 3 modelos base.
# Incluye funciones summarize_classification y build_model.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from cargar_datos import cargarDatos


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columnas_irrelevantes = ['id_cliente', 'nombre_cliente']
    df.drop(
        columns=[c for c in columnas_irrelevantes if c in df.columns],
        inplace=True, errors='ignore'
    )

    categorias_validas = ['Creciente', 'Decreciente', 'Estable']
    df['tendencia_ingresos'] = df['tendencia_ingresos'].where(
        df['tendencia_ingresos'].isin(categorias_validas), np.nan
    )

    df['salario_cero'] = (df['salario_cliente'] == 0).astype(int)
    df['salario_cliente'] = df['salario_cliente'].replace(0, np.nan)
    df['ratio_endeudamiento'] = df['saldo_total'] / df['salario_cliente']
    df['ratio_endeudamiento'] = df['ratio_endeudamiento'].replace([np.inf, -np.inf], np.nan)

    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'], errors='coerce')
        df['anio_prestamo'] = df['fecha_prestamo'].dt.year
        df['mes_prestamo'] = df['fecha_prestamo'].dt.month
        df['dia_semana'] = df['fecha_prestamo'].dt.dayofweek
        df.drop(columns=['fecha_prestamo'], inplace=True)

    if 'puntaje' in df.columns:
        df.drop(columns=['puntaje'], inplace=True)

    return df


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


def summarize_classification(y_test, y_pred, y_prob, nombre: str):
    print("\n" + "="*75)
    print(f" EVALUACION — {nombre}")
    print("="*75)
    print("\n Matriz de Confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Reporte de Clasificacion:")
    print(classification_report(y_test, y_pred, zero_division=0))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n ROC-AUC: {roc_auc:.4f}")
    return roc_auc


def build_model(model_class, X_train, y_train, **kwargs):
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluar_modelo(model, X_test, y_test, nombre: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = summarize_classification(y_test, y_pred, y_prob, nombre)

    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        'Modelo':      nombre,
        'ROC-AUC':     round(roc_auc, 4),
        'Accuracy':    round(reporte['accuracy'], 4),
        'Precision-0': round(reporte['0']['precision'], 4),
        'Recall-0':    round(reporte['0']['recall'], 4),
        'F1-0':        round(reporte['0']['f1-score'], 4),
        'Precision-1': round(reporte['1']['precision'], 4),
        'Recall-1':    round(reporte['1']['recall'], 4),
        'F1-1':        round(reporte['1']['f1-score'], 4),
    }


_CONFIG_MODELOS = [
    ('Logistic Regression', LogisticRegression, {
        'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42
    }),
    ('Random Forest', RandomForestClassifier, {
        'n_estimators': 300, 'class_weight': 'balanced',
        'max_depth': 10, 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1
    }),
    ('XGBoost', XGBClassifier, {
        'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
        'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1, 'verbosity': 0
    }),
]


def entrenar_y_evaluar(model_class, kwargs, nombre, X_train, y_train, X_test, y_test) -> tuple:
    if nombre == 'XGBoost':
        kwargs['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

    model = build_model(model_class, X_train, y_train, **kwargs)
    metricas = evaluar_modelo(model, X_test, y_test, nombre)
    return model, metricas


def comparar_modelos(lista_metricas: list) -> pd.DataFrame:
    df_comp = pd.DataFrame(lista_metricas)
    df_comp = df_comp.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    mejor = df_comp.iloc[0]['Modelo']

    print("\n" + "="*85)
    print(" TABLA COMPARATIVA DE MODELOS")
    print("="*85)
    print(df_comp.to_string(index=False))
    print(f"\n Mejor modelo por ROC-AUC: {mejor}")
    print("="*85)

    return df_comp


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
    for nombre, model_class, kwargs in _CONFIG_MODELOS:
        print(f"\n[LOG] Entrenando {nombre}...")
        model, metrica = entrenar_y_evaluar(
            model_class, kwargs.copy(), nombre,
            X_train_t, y_train, X_test_t, y_test
        )
        modelos[nombre.lower().replace(' ', '_')] = model
        metricas_lista.append(metrica)

    df_comparativa = comparar_modelos(metricas_lista)
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