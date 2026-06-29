# ARCHIVO: src/ft_engineering.py
# VERSIÓN: V1.1.3 (Avance 2 FINAL - PRODUCCIÓN)
# DESCRIPCIÓN:
# Pipeline completo de Feature Engineering + Baseline Model.
# Incluye limpieza, eliminación de variables irrelevantes,
# creación de features, control de leakage y evaluación.
# Refactorizado para separar la limpieza y el preprocesamiento
# en funciones exportables y reutilizables desde otros módulos.


# IMPORTAMOS LIBRERIAS
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# CARGA DE DATOS
from cargar_datos import cargarDatos


# ================================================================
# FUNCIONES EXPORTABLES
# ================================================================

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza y feature engineering sobre un DataFrame.

    Recibe un DataFrame crudo y devuelve uno listo para el pipeline
    de preprocesamiento. Es la función central del módulo: puede ser
    importada desde cualquier otro script (model_monitoring, app, etc.)
    para garantizar que todos los datos pasan por el mismo tratamiento.

    Parámetros:
        df : DataFrame crudo cargado desde cualquier fuente.

    Retorna:
        DataFrame limpio con las features creadas y variables irrelevantes
        eliminadas.
    """
    df = df.copy()

    # ------------------------------------------------------------
    # 1. ELIMINACIÓN DE VARIABLES IRRELEVANTES
    # ------------------------------------------------------------
    print("\n[LOG] Eliminando variables irrelevantes...")

    columnas_irrelevantes = [
        'id_cliente',
        'nombre_cliente'
    ]

    df.drop(
        columns=[c for c in columnas_irrelevantes if c in df.columns],
        inplace=True,
        errors='ignore'
    )

    print("[LOG] IDs eliminados (no aportan al modelo)")


    # ------------------------------------------------------------
    # 2. LIMPIEZA CATEGÓRICA
    # ------------------------------------------------------------
    print("\n[LOG] Validando variables categóricas...")

    categorias_validas = ['Creciente', 'Decreciente', 'Estable']

    antes = df['tendencia_ingresos'].isna().sum()

    df['tendencia_ingresos'] = df['tendencia_ingresos'].where(
        df['tendencia_ingresos'].isin(categorias_validas),
        np.nan
    )

    despues = df['tendencia_ingresos'].isna().sum()

    print(f"[LOG] tendencia_ingresos: {antes} → {despues} nulos")


    # ------------------------------------------------------------
    # 3. FEATURE ENGINEERING FINANCIERO
    # ------------------------------------------------------------
    print("\n[LOG] Feature Engineering...")

    # Indicador de salario cero antes de reemplazar
    df['salario_cero'] = (df['salario_cliente'] == 0).astype(int)

    # Convertir 0 → NaN para imputación correcta
    df['salario_cliente'] = df['salario_cliente'].replace(0, np.nan)

    # Ratio de endeudamiento robusto
    df['ratio_endeudamiento'] = df['saldo_total'] / df['salario_cliente']
    df['ratio_endeudamiento'] = df['ratio_endeudamiento'].replace(
        [np.inf, -np.inf], np.nan
    )

    print("[LOG] Features financieras creadas")


    # ------------------------------------------------------------
    # 4. FEATURE ENGINEERING TEMPORAL
    # ------------------------------------------------------------
    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'], errors='coerce')

        df['anio_prestamo'] = df['fecha_prestamo'].dt.year
        df['mes_prestamo'] = df['fecha_prestamo'].dt.month
        df['dia_semana'] = df['fecha_prestamo'].dt.dayofweek

        df.drop(columns=['fecha_prestamo'], inplace=True)

        print("[LOG] Features temporales creadas y cambio de tipo")


    # ------------------------------------------------------------
    # 5. DATA LEAKAGE CONTROL
    # ------------------------------------------------------------
    print("\n[LOG] Control de leakage...")

    if 'puntaje' in df.columns:
        df.drop(columns=['puntaje'], inplace=True)
        print("[LOG] puntaje eliminado (leakage potencial)")

    return df


def construir_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construye el ColumnTransformer de preprocesamiento según los
    tipos de variables presentes en X.

    Puede ser importado para reutilizar el mismo preprocessor en
    entrenamiento y en producción (por ejemplo, en model_monitoring).

    Parámetros:
        X : DataFrame de features ya limpio (sin target).

    Retorna:
        ColumnTransformer configurado pero aún no ajustado (sin fit).
    """
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    ordinal_features = ['tendencia_ingresos']

    cat_features = [
        c for c in X.select_dtypes(include=['object', 'category']).columns
        if c not in ordinal_features
    ]

    print(f"[LOG] Numéricas: {len(num_features)}")
    print(f"[LOG] Categóricas: {len(cat_features)}")
    print(f"[LOG] Ordinales: {len(ordinal_features)}")

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
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


# ================================================================
# FUNCIONES AUXILIARES DE MODELADO
# ================================================================

def summarize_classification(y_test, y_pred):
    print("\n" + "="*75)
    print(" RESUMEN DE EVALUACIÓN DEL MODELO")
    print("="*75)

    print("\n Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))


def build_model(model, X_train, y_train):
    print("\n[LOG] Entrenando modelo baseline...")
    model.fit(X_train, y_train)
    print("[LOG] Modelo entrenado correctamente")
    return model


# ================================================================
# PIPELINE PRINCIPAL
# ================================================================

def ft_engineering():

    print("\n" + "="*85)
    print(" PIPELINE FEATURE ENGINEERING + MODELADO BASELINE")
    print("="*85)


    # ------------------------------------------------------------
    # 1. CARGA DE DATOS
    # ------------------------------------------------------------
    df = cargarDatos().copy()

    print("\n[LOG] Dataset cargado")
    print(f"[LOG] Shape inicial: {df.shape}")


    # ------------------------------------------------------------
    # 2. LIMPIEZA Y FEATURE ENGINEERING
    # ------------------------------------------------------------
    df = preparar_datos(df)


    # ------------------------------------------------------------
    # 3. TARGET / FEATURES
    # ------------------------------------------------------------
    print("\n[LOG] Separando target y features...")

    target = 'Pago_atiempo'

    X = df.drop(columns=[target])
    y = df[target]

    print(f"[LOG] Features finales: {X.shape}")
    print(f"[LOG] Distribución target:\n{y.value_counts(normalize=True)}")


    # ------------------------------------------------------------
    # 4. TIPOS DE VARIABLES
    # ------------------------------------------------------------
    print("\n[LOG] Detectando tipos de variables...")

    preprocessor = construir_preprocessor(X)


    # ------------------------------------------------------------
    # 5. SPLIT TRAIN / TEST
    # ------------------------------------------------------------
    print("\n[LOG] Dividiendo train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("[LOG] Split completado")


    # ------------------------------------------------------------
    # 6. TRANSFORMACIÓN
    # ------------------------------------------------------------
    print("\n[LOG] Aplicando preprocessing...")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print(f"[LOG] Train shape: {X_train.shape}")
    print(f"[LOG] Test shape: {X_test.shape}")


    # ------------------------------------------------------------
    # 7. MODELO BASELINE
    # ------------------------------------------------------------
    print("\n[LOG] Entrenando modelo baseline...")

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    model = build_model(model, X_train, y_train)

    y_pred = model.predict(X_test)

    summarize_classification(y_test, y_pred)

    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n ROC-AUC: {roc:.4f}")

    print("\n PIPELINE FINALIZADO CORRECTAMENTE")


    return X_train, X_test, y_train, y_test, model


# ================================================================
# COMPROBACIÓN EJECUCIÓN
# ================================================================
if __name__ == "__main__":

    try:
        print("\n" + "="*85)
        print(" EJECUCIÓN LOCAL DEL PIPELINE")
        print("="*85)

        X_train_p, X_test_p, y_train_p, y_test_p, model = ft_engineering()

        print("\n PIPELINE EJECUTADO CORRECTAMENTE")
        print(" Sin errores en feature engineering ni modelado")

    except Exception as e:
        print("\n ERROR EN PIPELINE")
        print(f"Detalle: {e}")

