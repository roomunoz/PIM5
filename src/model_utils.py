# model_utils.py

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import pandas as pd
import numpy as np

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



def summarize_classification(y_test, y_pred, y_prob, nombre: str):
    print("\n" + "="*75)
    print(f" EVALUACION — {nombre}")
    print("="*75)

    print("\n Matriz de Confusion:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Reporte de Clasificacion:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n ROC-AUC: {roc_auc:.4f}")

    return roc_auc


