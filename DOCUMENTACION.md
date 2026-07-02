# PROYECTO INTEGRADOR M5
## Modelo de Riesgo Crediticio

**Prediccion de Pago a Tiempo mediante Aprendizaje Supervisado**

Data Science | Bootcamp Henry

| | |
|---|---|
| **Autor** | roomunoz |
| **Fecha** | Julio 2026 |

---

## Indice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Contexto del Negocio](#2-contexto-del-negocio)
3. [Descripcion del Dataset](#3-descripcion-del-dataset)
4. [Avance 1 — Analisis Exploratorio de Datos (EDA)](#4-avance-1--analisis-exploratorio-de-datos-eda)
5. [Avance 2 — Feature Engineering y Modelado Supervisado](#5-avance-2--feature-engineering-y-modelado-supervisado)
6. [Avance 3 — Monitoreo y Data Drift](#6-avance-3--monitoreo-y-data-drift)
7. [Avance 4 — Despliegue API con FastAPI y Docker](#7-avance-4--despliegue-api-con-fastapi-y-docker)
8. [Resultados y Metricas](#8-resultados-y-metricas)
9. [Conclusiones y Recomendaciones](#9-conclusiones-y-recomendaciones)
10. [Tecnologias Utilizadas](#10-tecnologias-utilizadas)

---

## 1. Resumen Ejecutivo

Este proyecto presenta el desarrollo completo de un modelo predictivo de riesgo crediticio utilizando tecnicas de Machine Learning supervisado. El objetivo es anticipar el comportamiento de pago de nuevos solicitantes de credito a partir de datos historicos de prestamos, permitiendo a la institucion financiera tomar decisiones informadas y reducir perdidas por mora.

Se implemento un pipeline completo de MLOps que abarca desde la carga y exploracion de datos, pasando por la ingenieria de caracteristicas, el entrenamiento y evaluacion de tres modelos (Regresion Logistica, Random Forest y XGBoost), hasta el despliegue en produccion mediante una API REST containerizada con Docker. Adicionalmente, se desarrollo un sistema de monitoreo de data drift y un dashboard interactivo para visualizacion continua.

El modelo seleccionado fue Regresion Logistica, priorizando el Recall de la clase minoritaria (mora) sobre metricas globales como ROC-AUC, alineandose con la estrategia de negocio de minimizar falsos negativos en la deteccion de morosos.

---

## 2. Contexto del Negocio

### 2.1 Problema

La empresa financiera enfrenta perdidas significativas debido a la morosidad en su cartera de creditos. El proceso actual de evaluacion de solicitudes es predominantemente manual y se basa en criterios subjetivos, lo que genera:

- Aprobacion de creditos a solicitantes que no podran pagar.
- Rechazo de solicitantes solventes por falta de informacion objetiva.
- Inconsistencia en las decisiones entre diferentes analistas.
- Alta tasa de mora en la cartera activa.

### 2.2 Solucion Propuesta

Desarrollar un modelo predictivo basado en datos historicos que automatice la evaluacion de riesgo crediticio, proporcionando una probabilidad de mora objetiva y reproducible para cada solicitante. El modelo se integra en un pipeline MLOps que garantiza su actualizacion continua y monitoreo constante.

### 2.3 Criterio de Exito

Dado que el dataset esta altamente desbalanceado (~95% paga a tiempo, ~5% mora), el criterio principal de exito es la capacidad del modelo para detectar clientes en mora (clase 0). Se prioriza el Recall de la clase 0 sobre el ROC-AUC, ya que el costo financiero de aprobar un prestamo a un moroso supera ampliamente el costo de oportunidad de rechazar a un cliente solvente.

---

## 3. Descripcion del Dataset

El dataset contiene informacion historica de ~10,700 creditos otorgados por la institucion financiera, con 23 variables que incluyen datos demograficos, financieros e historicos de credito.

### 3.1 Variables

| Variable | Tipo | No Nulos | Descripcion |
|---|---|---|---|
| tipo_credito | int64 | 10,763 | Codigo de tipo de credito (4-68) |
| fecha_prestamo | datetime64 | 10,763 | Fecha de otorgamiento del prestamo |
| capital_prestado | float64 | 10,763 | Monto del capital prestado |
| plazo_meses | int64 | 10,763 | Plazo en meses (2-90) |
| edad_cliente | int64 | 10,763 | Edad del cliente (19-123) |
| tipo_laboral | str | 10,763 | Empleado (62.8%) / Independiente (37.2%) |
| salario_cliente | int64 | 10,763 | Salario declarado (0-22B, outliers) |
| total_otros_prestamos | int64 | 10,763 | Total de otros prestamos activos |
| cuota_pactada | int64 | 10,763 | Cuota mensual acordada |
| puntaje | float64 | 10,763 | Puntaje interno (se descarta) |
| puntaje_datacredito | float64 | 10,757 | Puntaje de buro de credito |
| cant_creditosvigentes | int64 | 10,763 | Cantidad de creditos vigentes |
| huella_consulta | int64 | 10,763 | Numero de consultas a centrales |
| saldo_mora | float64 | 10,607 | Saldo en mora actual |
| saldo_total | float64 | 10,607 | Saldo total de la deuda |
| saldo_principal | float64 | 10,358 | Saldo de capital |
| saldo_mora_codeudor | float64 | 10,173 | Saldo en mora del codeudor |
| creditos_sectorFinanciero | int64 | 10,763 | Creditos en sector financiero |
| creditos_sectorCooperativo | int64 | 10,763 | Creditos en sector cooperativo |
| creditos_sectorReal | int64 | 10,763 | Creditos en sector real |
| promedio_ingresos_datacredito | float64 | 7,833 | Promedio de ingresos reportado |
| tendencia_ingresos | object | 7,831 | Tendencia: Creciente/Estable/Decreciente |
| Pago_atiempo | int64 | 10,763 | Target: 1=pago a tiempo, 0=mora |

### 3.2 Calidad de Datos

Durante el analisis exploratorio se identificaron los siguientes problemas de calidad:

- **tendencia_ingresos**: ~27% de valores nulos mas valores numericos invalidos mezclados con categorias. Se filtraron solo las categorias validas (Creciente, Estable, Decreciente).
- **salario_cliente**: Valores cero que representan datos faltantes. Se creo una variable indicadora (salario_cero) y se reemplazaron por NaN.
- **promedio_ingresos_datacredito**: ~27% de valores nulos, imputados con la mediana.
- **saldo_mora, saldo_principal, saldo_mora_codeudor**: Entre 1% y 6% de valores nulos, imputados con la mediana.

### 3.3 Desbalanceo de Clases

La variable target presenta un desbalanceo significativo:

| Clase | Significado | Proporcion |
|---|---|---|
| 1 (Paga a tiempo) | Buen pagador | 95.25% (10,252) |
| 0 (Mora) | Default | 4.75% (511) |

Para mitigar el desbalanceo se utilizaron: class_weight="balanced" en Regresion Logistica y Random Forest, y scale_pos_weight dinamico en XGBoost.

---

## 4. Avance 1 — Analisis Exploratorio de Datos (EDA)

### 4.1 Modulo de Carga de Datos

El archivo `cargar_datos.py` implementa la funcion `cargarDatos()` que lee el archivo Excel `Base_de_datos.xlsx` desde la raiz del proyecto, utilizando rutas relativas para garantizar portabilidad entre entornos.

### 4.2 Cuaderno EDA

El cuaderno `comprension_eda.ipynb` contiene el analisis exploratorio completo:

**4.2.1 Analisis Univariable**

Para cada variable numerica se analizaron: distribucion, medidas de tendencia central (media, mediana), dispersion (desviacion estandar, rango intercuartil), asimetria y presencia de outliers. Para las variables categoricas se analizaron frecuencias y proporciones.

**4.2.2 Analisis Bivariable**

Se generaron matrices de correlacion entre variables numericas, boxplots y graficos de densidad segmentados por el target para identificar variables con poder discriminante. Las variables mas correlacionadas con la mora fueron: puntaje_datacredito, tipo_credito, edad_cliente y ratio_endeudamiento.

**4.2.3 Analisis Multivariable**

Se realizaron pairplots de las variables mas relevantes y analisis de interacciones entre variables categoricas y numericas para identificar patrones complejos.

### 4.3 Transformaciones aplicadas (preparar_datos)

La funcion `preparar_datos()` en `model_utils.py` aplica las siguientes transformaciones:

- Eliminacion de columnas irrelevantes: id_cliente, nombre_cliente, puntaje.
- Filtrado de categorias validas en tendencia_ingresos: solo "Creciente", "Decreciente", "Estable".
- Creacion de salario_cero: variable binaria que indica si el salario es cero (dato faltante).
- Creacion de ratio_endeudamiento: saldo_total / salario_cliente, mide capacidad de pago.
- Parseo de fecha_prestamo: extraccion de anio, mes y dia de semana como variables numericas.

---

## 5. Avance 2 — Feature Engineering y Modelado Supervisado

### 5.1 Arquitectura

El pipeline de modelado se compone de tres modulos independientes pero coordinados:

- **model_utils.py**: Funciones de utilidad compartidas (preparar_datos, summarize_classification).
- **ft_engineering.py**: Construccion del preprocesador y orquestacion del pipeline completo.
- **model_training_evaluation.py**: Configuracion, entrenamiento y evaluacion de modelos.

### 5.2 Preprocesamiento (ColumnTransformer)

Se construyo un ColumnTransformer con tres pipelines de transformacion paralelos:

| Pipeline | Columnas | Transformaciones |
|---|---|---|
| Numerico | 23 variables numericas | SimpleImputer(median) + StandardScaler |
| Categorico | tipo_laboral | SimpleImputer(mode) + OneHotEncoder |
| Ordinal | tendencia_ingresos | SimpleImputer(mode) + OrdinalEncoder |

### 5.3 Modelos Entrenados

**5.3.1 Regresion Logistica**

Modelo lineal con class_weight="balanced" para compensar el desbalanceo. Hiperparametros: max_iter=1000, random_state=42. Ventaja: interpretabilidad y buen rendimiento en deteccion de la clase minoritaria.

**5.3.2 Random Forest**

Ensemble de 300 arboles de decision con class_weight="balanced", max_depth=10, min_samples_leaf=5. Ventaja: alta precision general y capacidad de capturar relaciones no lineales.

**5.3.3 XGBoost**

Gradient boosting con 300 estimadores, learning_rate=0.05, max_depth=6. Se aplico scale_pos_weight dinamico basado en la proporcion de clases. Ventaja: mejor ROC-AUC global.

### 5.4 Criterio de Seleccion

El pipeline selecciona el modelo con mejor **Recall-0** (deteccion de morosos) en lugar de ROC-AUC o Accuracy. Esto responde a una decision de negocio: el costo de no detectar a un moroso es significativamente mayor que el de rechazar a un cliente solvente.

### 5.5 Resultados Comparativos

| Modelo | ROC-AUC | Recall-0 | Precision-0 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.6439 | **0.5686** | 0.0733 | 0.6391 |
| Random Forest | 0.6688 | 0.1078 | **0.3929** | **0.9498** |
| XGBoost | **0.6707** | 0.2451 | 0.1445 | 0.8955 |

### 5.6 Modelo Seleccionado: Regresion Logistica

Regresion Logistica fue seleccionado por su Recall-0 de 0.5686, significativamente superior a Random Forest (0.1078) y XGBoost (0.2451). Esto significa que detecta al 56.9% de los morosos reales, mas del doble que XGBoost y mas de 5 veces que Random Forest.

Aunque su precision es baja (7.3%), en el contexto de riesgo crediticio es preferible tener mas falsos positivos (alertas de mora que resultan ser falsas) que falsos negativos (morosos no detectados). El modelo captura ~57 de cada 100 morosos reales, mientras que XGBoost captura solo ~25 y Random Forest ~11.

El modelo y el preprocesador ajustado se serializan con joblib y se almacenan en `models/modelo.pkl` y `models/preprocessor.pkl` respectivamente, listos para el despliegue.

---

## 6. Avance 3 — Monitoreo y Data Drift

### 6.1 Concepto de Data Drift

En produccion, la distribucion de los datos que recibe un modelo puede cambiar con el tiempo debido a factores economicos, cambios demograficos, nuevas politicas de credito, etc. Cuando la distribucion de las variables de entrada difiere significativamente de la distribucion original de entrenamiento, el rendimiento del modelo se degrada. Este fenomeno se conoce como Data Drift y su deteccion temprana es crucial para mantener la fiabilidad del sistema.

### 6.2 Metricas Implementadas

| Metrica | Tipo de Variable | Umbrales |
|---|---|---|
| Population Stability Index (PSI) | Numericas continuas y ordinales | <0.1: Sin cambio / 0.1-0.2: Moderado / >0.2: Drift |
| Chi-Cuadrado (X2) | Categoricas nominales | p < 0.05: Drift |

### 6.3 Modulo de Monitoreo

`model_monitoring.py` implementa la deteccion de data drift comparando el dataset original (`Base_de_datos.xlsx`) contra un dataset con drift simulado (`Base_de_datos_con_Data_Drift_Simulado.xlsx`).

El flujo de deteccion incluye:

- Carga de ambos datasets y aplicacion de preparar_datos().
- Aplicacion del preprocessor.pkl guardado para transformar ambos conjuntos.
- Calculo de PSI para cada variable numerica con division en 10 bins.
- Calculo de Chi2 para cada variable categorica.
- Asignacion de nivel de alerta: DRIFT, MODERADO o SIN CAMBIO.
- Generacion automatica de recomendaciones segun la severidad del drift.
- Exportacion de resultados a `models/reporte_drift.csv`.

Resultados del analisis de drift:

| Nivel de Alerta | Cantidad de Features |
|---|---|
| DRIFT | 7 |
| MODERADO | 5 |
| SIN CAMBIO | 13 |

Las variables con mayor drift fueron: puntaje_datacredito (PSI: 4.90), tipo_credito (PSI: 2.29) y edad_cliente (PSI: 1.74). Estos resultados indican que el modelo requiere reentrenamiento periodico para mantener su rendimiento.

### 6.4 Dashboard Interactivo

Se desarrollo un dashboard interactivo con **Streamlit** (`dashboard.py`) que permite la visualizacion y analisis del data drift en cuatro vistas:

- **Resumen de Drift**: Tarjetas con conteo de alertas, semaforo general, tabla completa de metricas con formato condicional, graficos de barras y recomendaciones automaticas.
- **Evolucion Temporal**: Analisis de PSI promedio a lo largo del tiempo agrupado por mes, con deteccion de tendencias crecientes en los ultimos 3 periodos.
- **Detalle por Feature**: Visualizacion comparativa (original vs nueva) para cada feature seleccionable, con histogramas para numericas y barras para categoricas.
- **Sobre el Proyecto**: Documentacion del caso de negocio, descripcion del dataset y tabla de referencia de metricas.

---

## 7. Avance 4 — Despliegue API con FastAPI y Docker

### 7.1 API REST con FastAPI

Se implemento una API REST utilizando FastAPI (`model_deploy.py`) que carga el modelo y el preprocesador serializados al iniciar y expone los siguientes endpoints:

| Endpoint | Metodo | Descripcion |
|---|---|---|
| /saludo | GET | Mensaje de bienvenida |
| /health | GET | Estado de carga del modelo y preprocesador |
| /predict | POST | Predice riesgo de mora para uno o mas clientes |

### 7.2 Esquema de Datos

El endpoint `/predict` acepta un array de objetos `Cliente` con todas las variables del dataset original (excepto el target `Pago_atiempo`). La API aplica el mismo pipeline de preprocesamiento (preparar_datos + ColumnTransformer) que se uso en entrenamiento, garantizando consistencia entre desarrollo y produccion.

La respuesta incluye para cada cliente: prediccion (0=paga a tiempo, 1=mora) y probabilidad de mora (entre 0 y 1).

### 7.3 Contenedor Docker

Se creo un Dockerfile que construye una imagen basada en python:3.11.6-slim, instala las dependencias del requirements.txt, copia el codigo fuente y los modelos, y ejecuta la API mediante uvicorn en el puerto 8000.

Comandos de despliegue:

```bash
docker build -t credit-risk-api .
docker run -d -p 8000:8000 --name credit-container credit-risk-api
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

---

## 8. Resultados y Metricas

### 8.1 Rendimiento del Modelo

El modelo de Regresion Logistica seleccionado presenta las siguientes metricas sobre el conjunto de test (20% del dataset, 2,153 registros):

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Mora) | 0.07 | 0.57 | 0.13 | 102 |
| 1 (Paga a tiempo) | 0.97 | 0.64 | 0.77 | 2,051 |

### 8.2 Matriz de Confusion

| | Predicho: 0 | Predicho: 1 |
|---|---|---|
| **Real: 0** | 58 (VP) | 44 (FN) |
| **Real: 1** | 733 (FP) | 1,318 (VN) |

VP = Verdaderos Positivos (morosos detectados), FN = Falsos Negativos (morosos no detectados), FP = Falsos Positivos (buenos pagadores marcados como morosos), VN = Verdaderos Negativos (buenos pagadores correctamente identificados).

### 8.3 Metricas Globales

| Metrica | Valor |
|---|---|
| ROC-AUC | 0.6439 |
| Accuracy | 0.6391 |
| Recall-0 (Mora) | 0.5686 |
| Recall-1 (Pago) | 0.6426 |

---

## 9. Conclusiones y Recomendaciones

### 9.1 Conclusiones

- Se desarrollo un pipeline completo de MLOps que integra carga, limpieza, modelado, despliegue y monitoreo, garantizando reproducibilidad y trazabilidad.
- Regresion Logistica fue seleccionada como modelo final por su capacidad para detectar morosos (Recall-0 = 56.9%), alineandose con la estrategia de negocio.
- La API REST containerizada permite integrar el modelo en sistemas existentes de manera escalable y portable.
- El sistema de monitoreo de data drift detecto 7 variables con drift significativo en el dataset simulado, validando la efectividad del mecanismo de alertas.
- El dashboard interactivo facilita la visualizacion continua del rendimiento del modelo y la deteccion temprana de degradacion.

### 9.2 Recomendaciones

- Reentrenar el modelo periodicamente (cada 3-6 meses) con datos actualizados para mantener su rendimiento ante cambios en la distribucion de los datos.
- Implementar un pipeline de CI/CD para automatizar el reentrenamiento y despliegue cuando se detecte drift significativo.
- Explorar tecnicas avanzadas de balanceo como SMOTE o undersampling para mejorar la deteccion de la clase minoritaria.
- Incorporar mas fuentes de datos (bureau extendido, redes sociales, etc.) para enriquecer el perfilamiento de los solicitantes.
- Configurar SonarCloud para el analisis continuo de calidad de codigo y seguridad.

---

## 10. Tecnologias Utilizadas

| Herramienta | Version | Uso |
|---|---|---|
| Python | 3.11.6 | Lenguaje principal |
| pandas | 3.0.3 | Manipulacion y analisis de datos |
| numpy | 2.4.6 | Computo numerico |
| scikit-learn | 1.9.0 | Preprocesamiento, modelos, metricas |
| XGBoost | 3.2.0 | Modelo gradient boosting |
| FastAPI | -- | Framework para API REST |
| Uvicorn | -- | Servidor ASGI |
| Streamlit | 1.58.0 | Dashboard interactivo |
| Docker | -- | Contenedorizacion |
| joblib | 1.5.3 | Serializacion de modelos |
| scipy | -- | Pruebas estadisticas |
| seaborn / matplotlib | -- | Visualizacion de datos |
| Git / GitHub | -- | Control de versiones |

---

> **Nota:** Este documento forma parte del Proyecto Integrador M5 del bootcamp en Data Science de Henry.
> **Autor:** roomunoz
