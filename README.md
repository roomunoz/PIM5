# PIM5 — Modelo de Riesgo Crediticio

Proyecto Integrador del bootcamp en Data Science (M5). Desarrollo de un modelo predictivo de **aprendizaje supervisado** para anticipar el comportamiento de pago de nuevos solicitantes de crédito, utilizando información histórica de préstamos. El proyecto sigue principios de **MLOps** para asegurar reproducibilidad, trazabilidad y escalabilidad.

---

## Tabla de Contenidos

- [Contexto del Proyecto](#contexto-del-proyecto)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Avance 1 — Carga de Datos y Análisis Exploratorio](#avance-1--carga-de-datos-y-análisis-exploratorio-eda)
- [Avance 2 — Feature Engineering y Modelado Supervisado](#avance-2--feature-engineering-y-modelado-supervisado-v110)
- [Avance 3 — Monitoreo y Data Drift](#avance-3--monitoreo-y-data-drift-v120)
- [Avance 4 — Despliegue API con FastAPI y Docker](#avance-4--despliegue-api-con-fastapi-y-docker)
- [Pipeline Completo](#pipeline-completo--flujo-de-datos)
- [Requisitos Técnicos](#requisitos-técnicos)
- [Configuración del Entorno](#configuración-del-entorno)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Licencia](#licencia)

---

## Contexto del Proyecto

### Problema de Negocio

Una empresa financiera necesita anticipar el comportamiento de pago de los solicitantes de crédito para minimizar pérdidas por mora. Actualmente, la evaluación de créditos se realiza mediante procesos manuales con criterios subjetivos, lo que genera:

- Alta tasa de mora en la cartera de créditos.
- Aprobación de solicitantes que no pueden pagar.
- Rechazo de solicitantes solventes por falta de información objetiva.

### Solución Propuesta

Desarrollar un modelo predictivo de **Machine Learning supervisado** que, basado en datos históricos de ~10,700 créditos con 23 variables, clasifique a los nuevos solicitantes en:

| Clase | Significado | Volumen en dataset |
|---|---|---|
| **0 — Mora** | No pagará a tiempo | ~4.75% (511 registros) |
| **1 — Pago a tiempo** | Pagará según lo acordado | ~95.25% (10,252 registros) |

### Estrategia de Modelado

Dado que el dataset está **altamente desbalanceado** (95% clase 1 vs 5% clase 0), el objetivo principal es **maximizar la detección de la clase minoritaria (mora)**. Se prioriza el **Recall de la clase 0** sobre el ROC-AUC, ya que el costo de no detectar a un moroso es mayor que el de rechazar a un cliente solvente.

---

## Estructura del Repositorio

### Ramas

El repositorio cuenta con tres ramas según la metodología del proyecto:

| Rama | Propósito |
|---|---|
| `master` | Versión estable en producción |
| `certification` | Versión certificada para revisión |
| `developer` | Desarrollo activo y experimentación |

### Directorios y Archivos

```
PIM5/
├── src/                                    # Código fuente del proyecto
│   ├── cargar_datos.py                     # Carga del dataset Excel
│   ├── comprension_eda.ipynb               # Análisis exploratorio (EDA)
│   ├── model_utils.py                      # Utilidades: limpieza y métricas
│   ├── ft_engineering.py                   # Feature Engineering + Pipeline de modelado
│   ├── model_training_evaluation.py        # Entrenamiento y evaluación de modelos
│   ├── model_monitoring.py                 # Monitoreo y detección de data drift
│   ├── dashboard.py                        # Dashboard Streamlit para monitoreo
│   └── model_deploy.py                     # API FastAPI para servir el modelo
├── models/                                 # Artefactos del modelo
│   ├── modelo.pkl                          # Mejor modelo entrenado (Logistic Regression)
│   ├── preprocessor.pkl                    # ColumnTransformer ajustado
│   └── reporte_drift.csv                   # Resultados de detección de drift
├── Base_de_datos.xlsx                      # Dataset histórico de créditos (original)
├── Base_de_datos_con_Data_Drift_Simulado.xlsx  # Dataset con drift simulado
├── requirements.txt                        # Dependencias del proyecto
├── Dockerfile                              # Contenedor para la API
├── .dockerignore                           # Exclusiones para Docker
├── .gitignore                              # Exclusiones para Git
├── LICENSE                                 # Licencia MIT
└── README.md                               # Documentación del proyecto
```

---

## Avance 1 — Carga de Datos y Análisis Exploratorio (EDA)

### `src/cargar_datos.py`

Módulo responsable de la carga del dataset `Base_de_datos.xlsx`. Utiliza rutas relativas basadas en la ubicación del archivo para garantizar portabilidad entre entornos de desarrollo y producción.

**Función principal:**

| Función | Descripción |
|---|---|
| `cargarDatos()` | Lee el archivo Excel y retorna un `DataFrame` de pandas con los ~10,763 registros |

### `src/comprension_eda.ipynb`

Cuaderno Jupyter con el análisis exploratorio completo del dataset. Contiene:

#### Exploración Inicial

| Aspecto | Resultado |
|---|---|
| **Dimensiones** | 10,763 filas, 23 columnas |
| **Tipos de datos** | 12 int64, 8 float64, 1 datetime64, 1 object, 1 str |
| **Target** | `Pago_atiempo` — 1 (paga a tiempo), 0 (mora) |
| **Desbalanceo** | 95.25% clase 1, 4.75% clase 0 |

#### Análisis Univariable

Para cada variable numérica se analizaron: distribución, medidas de tendencia central, dispersión, asimetría, outliers. Para las categóricas: frecuencias y proporciones.

Hallazgos clave:

| Variable | Problema detectado |
|---|---|
| `tendencia_ingresos` | ~27% de valores nulos + valores numéricos inválidos mezclados con categorías válidas ('Creciente', 'Decreciente', 'Estable') |
| `salario_cliente` | Outliers extremos hasta 22 mil millones; valores cero que representan datos faltantes |
| `promedio_ingresos_datacredito` | ~27% de valores nulos |
| `saldo_mora`, `saldo_principal`, `saldo_mora_codeudor` | Entre 1% y 6% de valores nulos |

#### Análisis Bivariable

- **Matriz de correlación** entre variables numéricas.
- **Relación de cada variable con el target** (boxplots para numéricas, barras apiladas para categóricas).
- Detección de variables con poder discriminante: `tipo_credito`, `puntaje_datacredito`, `edad_cliente`.

#### Análisis Multivariable

- Pairplots de las variables más correlacionadas con el target.
- Análisis de interacciones entre variables categóricas y numéricas.

### Decisiones de Limpieza

| Decisión | Justificación |
|---|---|
| Eliminar `id_cliente` y `nombre_cliente` | Sin valor predictivo |
| Eliminar `puntaje` | Variable interna sin significado claro |
| Filtrar `tendencia_ingresos` | Solo valores válidos ('Creciente', 'Decreciente', 'Estable'), el resto se trata como NaN |
| Crear `salario_cero` | Indicador binario de salario cero (dato faltante) |
| Crear `ratio_endeudamiento` | `saldo_total / salario_cliente`, mide capacidad de pago |
| Parsear `fecha_prestamo` | Extraer año, mes, día de semana como variables numéricas |

---

## Avance 2 — Feature Engineering y Modelado Supervisado (V1.1.0)

### Arquitectura del Módulo

```
src/
├── model_utils.py                  # Funciones reutilizables
├── ft_engineering.py               # Preprocesamiento + orquestación del pipeline
└── model_training_evaluation.py    # Construcción, entrenamiento y evaluación de modelos
```

### `src/model_utils.py` — Utilidades

| Función | Descripción |
|---|---|
| `preparar_datos(df)` | Limpieza y feature engineering ligero sobre el dataset crudo. Elimina columnas irrelevantes, filtra categorías inválidas, crea `salario_cero`, `ratio_endeudamiento`, y parsea `fecha_prestamo` en componentes temporales |
| `summarize_classification(y_test, y_pred, y_prob, nombre)` | Imprime matriz de confusión, reporte de clasificación y métrica ROC-AUC |

### `src/ft_engineering.py` — Feature Engineering

Construye un `ColumnTransformer` con tres pipelines de transformación:

| Pipeline | Columnas | Transformaciones |
|---|---|---|
| **Numérico** | 23 columnas numéricas (ingresos, deudas, puntajes, fechas parseadas, etc.) | `SimpleImputer(strategy='median')` + `StandardScaler` |
| **Categórico** | `tipo_laboral` ('Empleado', 'Independiente') | `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` |
| **Ordinal** | `tendencia_ingresos` ('Decreciente', 'Estable', 'Creciente') | `SimpleImputer(strategy='most_frequent')` + `OrdinalEncoder(categories=[['Decreciente','Estable','Creciente']])` |

**Función `ft_engineering()`:** Orquesta el pipeline completo:

1. Carga los datos con `cargarDatos()`.
2. Aplica `preparar_datos()` para limpieza inicial.
3. Separa features (`X`) y target (`y`).
4. Divide en train/test (80/20) con estratificación por target.
5. Construye y ajusta el `ColumnTransformer`.
6. Transforma train y test.
7. Entrena los 3 modelos.
8. Compara resultados y guarda el mejor modelo + preprocesador.

### `src/model_training_evaluation.py` — Modelos

#### Configuración de Modelos

| Modelo | Hiperparámetros | Manejo de desbalanceo |
|---|---|---|
| **Logistic Regression** | `max_iter=1000`, `random_state=42` | `class_weight='balanced'` |
| **Random Forest** | `n_estimators=300`, `max_depth=10`, `min_samples_leaf=5` | `class_weight='balanced'` |
| **XGBoost** | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` | `scale_pos_weight` dinámico |

#### Funciones

| Función | Descripción |
|---|---|
| `build_model(model_class, X_train, y_train, **kwargs)` | Instancia y entrena un clasificador |
| `evaluar_modelo(model, X_test, y_test, nombre)` | Evalúa el modelo y retorna métricas detalladas (ROC-AUC, Accuracy, Precision/Recall/F1 por clase) |
| `entrenar_y_evaluar(model_class, kwargs, nombre, X_train, y_train, X_test, y_test)` | Calcula `scale_pos_weight` para XGBoost, entrena y evalúa |
| `comparar_modelos(lista_metricas)` | Ordena modelos por Recall-0 descendente, imprime tabla comparativa |

### Selección del Modelo: Logistic Regression

#### Resultados Comparativos

| Modelo | ROC-AUC | Recall-0 (mora) | Precision-0 | F1-0 | Accuracy |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.6439 | **0.5686** | 0.0733 | 0.1299 | 0.6391 |
| XGBoost | **0.6707** | 0.2451 | 0.1445 | 0.1818 | 0.8955 |
| Random Forest | 0.6688 | 0.1078 | **0.3929** | **0.1692** | **0.9498** |

#### Justificación de Negocio

Aunque XGBoost y Random Forest superan a Logistic Regression en ROC-AUC y accuracy global, **el objetivo del negocio no es maximizar la precisión general, sino minimizar las pérdidas por mora**.

- **Logistic Regression detecta al 56.9% de los morosos**, más del doble que XGBoost (24.5%) y más de 5 veces que Random Forest (10.8%).
- El costo de **aprobar un préstamo a un moroso** (pérdida total del capital) es significativamente mayor que el de **rechazar a un cliente solvente** (costo de oportunidad).
- Con un Recall-0 de 0.57, Logistic Regression captura ~57 de cada 100 morosos reales, frente a solo ~11 de Random Forest o ~25 de XGBoost.

Por esta razón, el pipeline selecciona el modelo con mejor **Recall-0** en lugar de ROC-AUC.

### Ejecución

```bash
cd PIM5
.\venv\Scripts\activate
python src/ft_engineering.py
```

---

## Avance 3 — Monitoreo y Data Drift (V1.2.0)

### ¿Qué es Data Drift?

En producción, los datos que recibe un modelo pueden cambiar con el tiempo debido a factores económicos, cambios en el perfil de clientes, nuevas políticas de crédito, etc. Cuando la distribución de las variables de entrada difiere significativamente de la distribución con la que fue entrenado el modelo, su rendimiento se degrada. Este fenómeno se conoce como **Data Drift**.

### Sistema de Monitoreo

#### `src/model_monitoring.py`

Script de detección de data drift que compara el dataset original (`Base_de_datos.xlsx`) contra un dataset con drift simulado (`Base_de_datos_con_Data_Drift_Simulado.xlsx`).

**Métricas implementadas:**

| Métrica | Tipo de Variable | Interpretación |
|---|---|---|
| **Population Stability Index (PSI)** | Numéricas | `< 0.1`: Sin cambio — `0.1–0.2`: Moderado — `> 0.2`: Drift |
| **Chi-Cuadrado (χ²)** | Categóricas | `p < 0.05`: Evidencia de drift |

**Flujo de detección:**

1. Carga ambos datasets (original y con drift).
2. Aplica `preparar_datos()` a ambos.
3. Transforma con el `preprocessor.pkl` guardado.
4. Calcula PSI para cada variable numérica.
5. Calcula Chi² para cada variable categórica.
6. Asigna nivel de alerta (DRIFT / MODERADO / SIN CAMBIO).
7. Genera recomendaciones automáticas según severidad.
8. Exporta resultados a `models/reporte_drift.csv`.

**Resultados del análisis:**

| Nivel de Alerta | Cantidad de Features |
|---|---|
| **DRIFT** | 7 features |
| **MODERADO** | 5 features |
| **SIN CAMBIO** | 13 features |

Las variables con mayor drift detectado fueron: `puntaje_datacredito` (PSI: 4.90), `tipo_credito` (PSI: 2.29), `edad_cliente` (PSI: 1.74).

#### `src/dashboard.py` — Dashboard de Monitoreo

Dashboard interactivo construido con **Streamlit** para visualización y análisis del data drift.

**Páginas:**

| Página | Descripción |
|---|---|
| **Sobre el Proyecto** | Documentación del caso de negocio, descripción del dataset y tabla de referencia de métricas |
| **Resumen de Drift** | Tarjetas con conteo de alertas, semáforo general, tabla completa de métricas con formato condicional, gráficos de barras y recomendaciones automáticas |
| **Evolución Temporal** | Análisis de PSI promedio a lo largo del tiempo (agrupado por mes), detección de tendencias crecientes en los últimos 3 períodos |
| **Detalle por Feature** | Visualización comparativa (original vs nueva) para cada feature seleccionable: histogramas para numéricas, barras para categóricas, con métricas en vivo |

### Ejecución

```bash
# Detección de drift en consola
python src/model_monitoring.py

# Dashboard Streamlit
streamlit run src/dashboard.py
```

---

## Avance 4 — Despliegue API con FastAPI y Docker

### `src/model_deploy.py` — API REST

API construida con **FastAPI** que expone el modelo Logistic Regression en producción. Carga los artefactos `modelo.pkl` y `preprocessor.pkl` al iniciar.

**Endpoints:**

| Endpoint | Método | Descripción |
|---|---|---|
| `/saludo` | GET | Mensaje de bienvenida de la API |
| `/health` | GET | Estado de carga del modelo y preprocesador |
| `/predict` | POST | Predice riesgo de mora para uno o más clientes |

#### Esquema de Entrada — POST `/predict`

```json
{
  "clientes": [
    {
      "tipo_credito": 7,
      "fecha_prestamo": "2024-12-21",
      "capital_prestado": 3692160.0,
      "plazo_meses": 10,
      "edad_cliente": 42,
      "tipo_laboral": "Independiente",
      "salario_cliente": 8000000,
      "total_otros_prestamos": 2500000,
      "cuota_pactada": 341296,
      "puntaje_datacredito": 695.0,
      "cant_creditosvigentes": 10,
      "huella_consulta": 5,
      "saldo_mora": 0.0,
      "saldo_total": 51258.0,
      "saldo_principal": 51258.0,
      "saldo_mora_codeudor": 0.0,
      "creditos_sectorFinanciero": 5,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 0,
      "promedio_ingresos_datacredito": 908526.0,
      "tendencia_ingresos": "Estable"
    }
  ]
}
```

**Campos opcionales:** `id_cliente`, `nombre_cliente`, `puntaje` (se descartan internamente).

#### Esquema de Salida

```json
[
  {
    "prediccion": 0,
    "probabilidad_mora": 0.3487
  }
]
```

| Campo | Tipo | Descripción |
|---|---|---|
| `prediccion` | int | `0` = pagará a tiempo, `1` = entrará en mora |
| `probabilidad_mora` | float | Probabilidad de mora (clase 1), entre 0 y 1 |

### `Dockerfile`

```dockerfile
FROM python:3.11.6-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Despliegue

#### Con Docker

```bash
# Construir imagen
cd PIM5
docker build -t credit-risk-api .

# Ejecutar contenedor en segundo plano
docker run -d -p 8000:8000 --name credit-container credit-risk-api

# Verificar estado
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

# Probar predicción (PowerShell)
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post `
  -Body '{"clientes":[{"tipo_credito":7,"fecha_prestamo":"2024-12-21","capital_prestado":3692160.0,"plazo_meses":10,"edad_cliente":42,"tipo_laboral":"Independiente","salario_cliente":8000000,"total_otros_prestamos":2500000,"cuota_pactada":341296,"puntaje_datacredito":695.0,"cant_creditosvigentes":10,"huella_consulta":5,"saldo_mora":0.0,"saldo_total":51258.0,"saldo_principal":51258.0,"saldo_mora_codeudor":0.0,"creditos_sectorFinanciero":5,"creditos_sectorCooperativo":0,"creditos_sectorReal":0,"promedio_ingresos_datacredito":908526.0,"tendencia_ingresos":"Estable"}]}' `
  -ContentType "application/json"

# Detener y limpiar
docker stop credit-container
docker rm credit-container
```

#### Sin Docker (directo con Python)

```bash
cd PIM5
.\venv\Scripts\activate
uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000 --reload
```

---

## Pipeline Completo — Flujo de Datos

```
Base_de_datos.xlsx
        │
        ▼
┌───────────────────┐
│  cargar_datos()   │
└────────┬──────────┘
         │
         ▼
┌──────────────────────┐
│  preparar_datos()    │  ← model_utils.py
│  (limpieza +         │
│   feature creation)  │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────┐
│   ft_engineering.py              │
│                                  │
│   ├─ construir_preprocessor()    │
│   │  (ColumnTransformer:         │
│   │   num + cat + ord)           │
│   │                              │
│   ├─ train_test_split(80/20)     │
│   │  (estratificado por target)  │
│   │                              │
│   ├─ fit_transform(X_train)      │
│   │  transform(X_test)           │
│   │                              │
│   ├─ entrenar 3 modelos          │
│   │  (LR, RF, XGBoost)           │
│   │                              │
│   ├─ comparar por Recall-0       │
│   │                              │
│   └─ guardar modelo.pkl          │
│      + preprocessor.pkl          │
└────────┬─────────────────────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌────────────────────┐          ┌──────────────────────┐
│  model_deploy.py   │          │  model_monitoring.py │
│  (FastAPI API)     │          │  (Data Drift)        │
│                    │          │                      │
│  GET  /saludo      │          │  PSI + Chi²          │
│  GET  /health      │          │  reporte_drift.csv   │
│  POST /predict     │          │                      │
└────────┬───────────┘          └──────────┬───────────┘
         │                                 │
         ▼                                 ▼
┌────────────────┐               ┌──────────────────┐
│   Dockerfile   │               │  dashboard.py    │
│  (container)   │               │  (Dashboard)     │
└────────────────┘               └──────────────────┘
```

---

## Requisitos Técnicos

### Dependencias (`requirements.txt`)

```
pandas==3.0.3
numpy==2.4.6
openpyxl
python-dotenv
jupyter
notebook
ipykernel
seaborn
ipywidgets
scikit-learn==1.9.0
xgboost==3.2.0
pydantic
fastapi
joblib==1.5.3
scipy
streamlit
uvicorn
```

### Configuración del Entorno

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd PIM5

# 2. Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\activate      # Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Ejecutar pipeline de entrenamiento
python src/ft_engineering.py
```

---

## Tecnologías Utilizadas

| Herramienta | Versión | Uso |
|---|---|---|
| **Python** | 3.11.6 | Lenguaje principal |
| **pandas** | 3.0.3 | Manipulación y análisis de datos |
| **numpy** | 2.4.6 | Cómputo numérico |
| **scikit-learn** | 1.9.0 | Preprocesamiento, modelos, métricas |
| **XGBoost** | 3.2.0 | Modelo gradient boosting |
| **FastAPI** | — | Framework para API REST |
| **Uvicorn** | — | Servidor ASGI |
| **Streamlit** | — | Dashboard interactivo de monitoreo |
| **Docker** | — | Contenedorización de la API |
| **joblib** | 1.5.3 | Serialización de modelos |
| **scipy** | — | Pruebas estadísticas (Chi², KS) |
| **seaborn** | — | Visualización |
| **matplotlib** | — | Visualización |
| **Jupyter** | — | Notebooks de EDA |
| **Git** | — | Control de versiones |
| **GitHub** | — | Repositorio remoto |

---

## Licencia

MIT License — Copyright (c) 2026 roomunoz

Ver archivo [LICENSE](LICENSE) para más detalles.
