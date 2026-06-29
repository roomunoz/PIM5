# PIM5 — Modelo de Riesgo Crediticio

## Descripción del Proyecto

Proyecto Integrador del bootcamp en Data Science. Se desarrolla un modelo predictivo de **aprendizaje supervisado** para anticipar el comportamiento de pago de nuevos solicitantes de crédito, utilizando información histórica de préstamos. El proyecto sigue principios de **MLOps** para asegurar reproducibilidad, trazabilidad y escalabilidad.

El dataset contiene ~10,700 registros con 23 variables, incluyendo información demográfica, financiera e histórica de créditos.

---

## Estructura del Repositorio

```
PIM5/
├── src/
│   ├── cargar_datos.py              # Carga del dataset Excel
│   ├── comprension_eda.ipynb        # Análisis exploratorio (EDA)
│   ├── ft_engineering.py            # Feature Engineering + Pipeline de modelado
│   ├── model_training_evaluation.py # Entrenamiento y evaluación de modelos
│   ├── model_utils.py               # Utilidades: limpieza, métricas
│   ├── app.py                     # Dashboard Streamlit (Avance 3)
│   ├── model_deploy.py              # API FastAPI para servir el modelo
│   └── model_monitoring.py          # Monitoreo y detección de data drift
├── Base_de_datos.xlsx               # Dataset histórico de créditos
├── requirements.txt                 # Dependencias del proyecto
├── .gitignore
├── LICENSE
└── README.md
```

---

## Avance 2 — Feature Engineering y Modelado Supervisado (V1.1.0)

### Arquitectura del Código

El pipeline de Avance 2 está dividido en **tres módulos independientes**, cada uno con una responsabilidad específica:

```
src/
├── model_utils.py                  # Módulo de utilidades: funciones reutilizables
├── ft_engineering.py               # Preprocesamiento + Feature Engineering + Orquestación del pipeline
└── model_training_evaluation.py    # Construcción, entrenamiento y evaluación de modelos
```

### 1. `model_utils.py` — Módulo de utilidades

Agrupa funciones auxiliares que se usan desde distintos módulos del proyecto, evitando repetir código y manteniendo todo ordenado:

| Función | Se usa desde | Qué hace |
|---|---|---|
| `preparar_datos(df)` | `ft_engineering.py`, `model_monitoring.py` | Limpieza inicial del dataset: elimina `id_cliente`, `nombre_cliente`, `puntaje`; filtra valores inválidos en `tendencia_ingresos`; crea `salario_cero` y `ratio_endeudamiento`; parsea `fecha_prestamo` a año/mes/día_semana |
| `summarize_classification(y_test, y_pred, y_prob, nombre)` | `model_training_evaluation.py` | Imprime matriz de confusión, reporte de clasificación y métrica ROC-AUC |

### 2. `ft_engineering.py` — Feature Engineering (V1.1.0)

Se encarga del **preprocesamiento y transformación de variables**, y **orquesta todo el pipeline**:

| Componente | Detalle |
|---|---|
| `construir_preprocessor(X)` | Crea un `ColumnTransformer` con 3 pipelines internos: |
| ├─ **Pipeline numérico** | Imputación por **mediana** + `StandardScaler` |
| ├─ **Pipeline categórico** | Imputación por **moda** + `OneHotEncoder` (con `handle_unknown='ignore'`) |
| └─ **Pipeline ordinal** | Imputación por **moda** + `OrdinalEncoder` con orden `[Decreciente, Estable, Creciente]` |
| `ft_engineering()` | Función principal: carga datos → `preparar_datos()` → separa X/y → train/test split (80/20, estratificado) → transforma con preprocessor → entrena los 3 modelos → compara resultados |

### 3. `model_training_evaluation.py` — Model Training & Evaluation

Módulo especializado en **construir, entrenar y evaluar modelos supervisados**:

| Función | Responsabilidad |
|---|---|
| `CONFIG_MODELOS` | Configuración de los 3 modelos con hiperparámetros definidos |
| `build_model(model_class, X_train, y_train, **kwargs)` | Instancia y entrena un modelo dado |
| `evaluar_modelo(model, X_test, y_test, nombre)` | Predice y calcula **ROC-AUC, Accuracy, Precision, Recall, F1** por clase (0 y 1) |
| `entrenar_y_evaluar(model_class, kwargs, nombre, ...)` | Agrega `scale_pos_weight` dinámico para XGBoost, luego entrena y evalúa |
| `comparar_modelos(lista_metricas)` | Ordena modelos por ROC-AUC descendente e imprime tabla comparativa |

**Los 3 modelos entrenados:**

| Modelo | Hiperparámetros clave | Manejo de desbalanceo |
|---|---|---|
| **Logistic Regression** | `max_iter=1000`, `random_state=42` | `class_weight='balanced'` |
| **Random Forest** | `n_estimators=300`, `max_depth=10`, `min_samples_leaf=5` | `class_weight='balanced'` |
| **XGBoost** | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` | `scale_pos_weight` dinámico |

### Flujo de Ejecución (dentro de `ft_engineering()`)

```
1. cargarDatos()
         │
         ▼
2. preparar_datos()          ← model_utils.py
         │
         ▼
3. construir_preprocessor(X)  ← define ColumnTransformer
         │
         ▼
4. train_test_split(80/20, stratify)
         │
         ▼
5. preprocessor.fit_transform(X_train)
   preprocessor.transform(X_test)
         │
         ▼
6. entrenar_y_evaluar() x 3  ← model_training_evaluation.py
   (LogisticRegression, RandomForest, XGBoost)
         │
         ▼
7. comparar_modelos()         ← ordena por ROC-AUC
```

### Cómo ejecutar

```bash
# 1. Activar entorno virtual
venv\Scripts\activate

# 2. Ejecutar pipeline completo (orquestado desde ft_engineering.py)
python src/ft_engineering.py
```

---

## Avance 3 — Monitoreo y Data Drift (V1.2.0)

### ¿Qué es Data Drift?

Cuando un modelo se entrena con datos históricos y recibe datos nuevos con distribuciones distintas (cambios económicos, nuevos perfiles de clientes, etc.), el modelo empieza a fallar porque opera en un terreno que no conoce. **Data drift** es la detección de esos cambios en las variables de entrada.

### Métricas utilizadas

| Métrica | Tipo de variable | Qué mide | Umbrales |
|---|---|---|---|
| **Population Stability Index (PSI)** | Numéricas continuas y ordinales | Cambio en distribución por bins | < 0.1: sin cambio. 0.1-0.2: moderado. > 0.2: drift |
| **Kolmogorov-Smirnov (KS Test)** | Numéricas continuas | Distancia máxima entre funciones de distribución acumulada | p > 0.05: sin cambio. 0.01-0.05: moderado. < 0.01: drift |
| **Jensen-Shannon Divergence (JSD)** | Numéricas continuas | Divergencia simétrica entre distribuciones (0 a 1) | < 0.05: sin cambio. 0.05-0.15: moderado. > 0.15: drift |
| **Chi-Cuadrado (χ²)** | Categóricas nominales | Diferencia en frecuencias observadas vs esperadas | p_valor < 0.05: drift |

### Módulos

#### `model_monitoring.py`
Script de detección de data drift. Compara el dataset original (`Base_de_datos.xlsx`) contra el nuevo dataset con drift simulado (`Base_de_datos_con_Data_Drift_Simulado.xlsx`):
1. Carga y prepara ambos datasets con `preparar_datos()`
2. Aplica el `preprocessor.pkl` guardado para transformar ambos por igual
3. Calcula **PSI + KS + JSD** para cada feature numérica y **Chi-Cuadrado** para cada categórica
4. Asigna alerta combinada (la peor de las 3 métricas numéricas)
5. Genera **recomendaciones automáticas** según la cantidad de features con drift
6. Reporte en consola con tabla completa y semáforo

#### `app.py`
Dashboard interactivo en **Streamlit** con cinco vistas:
- **Resumen de Drift**: cards con conteo de alertas, tabla con todas las métricas, gráficos de PSI y JSD, recomendaciones
- **Evolución Temporal**: agrupa por mes usando `fecha_prestamo` y grafica PSI/JSD promedio a lo largo del tiempo, con detección de tendencias crecientes
- **Detalle por Feature**: histogramas (numéricas) o barras (categóricas) comparando distribución original vs nueva, con métricas en vivo
- **Alertas y Recomendaciones**: semáforo general del modelo, recomendaciones automáticas, listado de features en DRIFT y MODERADO con sugerencias de acción
- **Sobre el Proyecto**: documentación del caso de negocio y tabla de referencia de métricas

### Cómo ejecutar

```bash
# Detección de drift en consola
python src/model_monitoring.py

# Dashboard Streamlit
streamlit run src/app.py
```

| Herramienta | Uso |
|---|---|
| **Python 3.x** | Lenguaje principal |
| **pandas / numpy** | Manipulación y análisis de datos |
| **scikit-learn** | Preprocesamiento, modelos, métricas |
| **XGBoost** | Modelo gradient boosting |
| **seaborn / matplotlib** | Visualización (EDA) |
| **FastAPI** | API de despliegue (Avance 4) |
| **Streamlit** | Dashboard interactivo (Avance 3) |
| **Docker** | Contenedorización (Avance 4) |

---

## Próximos Pasos

- **Avance 4:** API con FastAPI, imagen Docker, `model_deploy.py`

---

## Autor

**roomunoz** — Proyecto Integrador M5 | Data Science
