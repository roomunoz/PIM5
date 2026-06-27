# librerías
import pandas as pd
from cargar_datos import cargarDatos
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# cargar los datos
df = cargarDatos()

# vamos a tener una vista previa de los datos
print(df.head())
print(df.info())
print(df.describe())

# Paso 1: features/target split
X = df.drop('Pago_atiempo', axis=1) # features
y = df['Pago_atiempo']             # target

# Paso 2: definir variables por tipo
num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes('object').columns

print("Numeric features")
print(num_features)
print("Categorical features")
print(cat_features)

# Paso 3: Crear pipelines para cada ruta
## Ruta 1: numéricas
num_transformer =  Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='mean'))
])

## Ruta 2: categóricas
cat_transformer = Pipeline(steps=[
    ('to_str', FunctionTransformer(lambda x: x.astype(str))),
    ('inputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Paso 4: Combinar las 2 rutas en ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# Paso 5: dividir el dataset en train/test (antes de preprocesar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Paso 6: Aplicamos el preprocesamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Paso 7: resultados del preprocesamiento 
print("X_train preprocesados:")
print(X_train_processed)
print(X_train_processed.shape)
print("X_test preprocesados:")
print(X_test_processed)
print(X_test_processed.shape)

# Paso 8: construimos una función para "exportar": ft_engineering()
def ft_engineering():
    num_features = X.select_dtypes('number').columns
    cat_features = X.select_dtypes('object').columns
    num_transformer =  Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='mean'))
    ])

    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('inputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
            ]
    )

    return preprocessor