
from cargar_datos import cargarDatos
from ft_engineering import construir_preprocessor, preparar_datos


def nuevaFuncion():
    df = cargarDatos()
    df_ref = preparar_datos(df)
    preprocessor = construir_preprocessor(df)