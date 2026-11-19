# 🩺 Sistema de predicción de cáncer de mama con Machine Learning

Proyecto final de **Proyectos de Machine Learning**, período 2026-2.
Autora: **Cristina Soledad Proaño Ullaguari**

Es una aplicación en Python con **backend (FastAPI)** y **frontend (Streamlit)** que entrena modelos de clasificación para estimar si un tumor de mama es **benigno** o **maligno** a partir de 30 medidas de núcleos celulares sacadas de imágenes de biopsias por aspiración con aguja fina (FNA).

> ⚠️ Es un proyecto educativo. No reemplaza el diagnóstico de un profesional de la salud.

## Funcionalidades

| Requisito del proyecto | Dónde está |
|---|---|
| Entrenar el modelo | Página **Entrenar modelo** → `POST /entrenar` |
| Crear un API con el modelo e implementarlo | `api/main.py` (FastAPI, documentación en `/docs`) |
| Configurar parámetros | Algoritmo, hiperparámetros, % de prueba, folds de validación cruzada, semilla, estandarización y umbral de decisión |
| Mostrar métricas del entrenamiento | Exactitud, precisión, sensibilidad, especificidad, F1, ROC AUC, matriz de confusión, curva ROC, validación cruzada e importancia de características |
| Recibir un nuevo caso | Página **Nuevo caso** → `POST /predecir` |
| Limpieza de datos | `ml/datos.py`: duplicados, diagnósticos inválidos, valores negativos, nulos (imputación por mediana) y detección de atípicos (IQR) |

Algoritmos disponibles: **Regresión Logística, Random Forest, SVM y KNN**.

## Dataset (fuente real)

Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

- 569 casos (357 benignos, 212 malignos), 30 características numéricas.
- El archivo `data/breast_cancer_wisconsin.csv` se generó con `scripts/exportar_dataset.py` a partir de la copia del archivo original `wdbc.data` que trae scikit-learn. Así la app funciona sin internet.

## Estructura

```
proyecto-ml-cancer-mama/
├── api/main.py                 # Backend: API REST con FastAPI
├── frontend/app.py             # Frontend: interfaz gráfica con Streamlit
├── ml/
│   ├── datos.py                # Carga y limpieza de datos
│   └── entrenamiento.py        # Entrenamiento, métricas, guardado y predicción
├── data/breast_cancer_wisconsin.csv
├── modelos/                    # Aquí se guarda el modelo entrenado (.joblib)
├── scripts/exportar_dataset.py
├── tests/test_api.py           # Pruebas automáticas (pytest)
└── requirements.txt
```

## Cómo ejecutarlo

Requisitos: Python 3.10 o superior.

```bash
# 1. Crear y activar un entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Levantar el backend (terminal 1)
uvicorn api.main:app --reload
#   -> API en http://localhost:8000  |  documentación en http://localhost:8000/docs

# 4. Levantar el frontend (terminal 2, con el entorno activado)
streamlit run frontend/app.py
#   -> se abre en http://localhost:8501

# 5. (Opcional) correr las pruebas
pytest -q
```

Comandos desde la carpeta raíz del proyecto.

## Endpoints del API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/salud` | Estado del API y si hay modelo entrenado |
| GET | `/dataset/resumen` | Estadísticas y reporte de limpieza |
| GET | `/dataset/ejemplo?tipo=M\|B` | Un caso real para probar |
| GET | `/algoritmos` | Algoritmos y parámetros configurables |
| POST | `/entrenar` | Entrena y guarda un modelo |
| GET | `/metricas` | Métricas del modelo activo |
| GET | `/historial` | Entrenamientos anteriores |
| POST | `/predecir` | Predice un nuevo caso |

Ejemplo:

```bash
curl -X POST http://localhost:8000/entrenar -H "Content-Type: application/json" \
  -d '{"algoritmo": "svm", "parametros": {"C": 1.0, "kernel": "rbf"}, "test_size": 0.2}'
```

## Resultados de referencia

Con la configuración por defecto (20% prueba, semilla 42, datos estandarizados):

| Modelo | Exactitud | Sensibilidad | Especificidad | F1 | ROC AUC |
|---|---|---|---|---|---|
| Regresión Logística | 96.5% | 92.9% | 98.6% | 95.1% | 0.996 |
| Random Forest | 96.5% | 90.5% | 100% | 95.0% | 0.994 |
| **SVM (rbf)** | **97.4%** | **95.2%** | 98.6% | **96.4%** | 0.995 |
| KNN (k=5) | 95.6% | 90.5% | 98.6% | 93.8% | 0.982 |

## Referencias

- Street, W. N., Wolberg, W. H. y Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *Proc. SPIE 1905, Biomedical Image Processing and Biomedical Visualization*, 861–870.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Cortes, C. y Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
