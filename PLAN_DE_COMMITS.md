# Plan de commits progresivos

El profesor va a revisar que haya commits progresivos. No subas todo en un solo commit: ve agregando los archivos en este orden, idealmente en días distintos. (Puedes borrar este archivo antes del último commit.)

```bash
git init
git branch -M main
git remote add origin https://github.com/TU_USUARIO/proyecto-ml-cancer-mama.git
```

| # | Qué agregar | Comando |
|---|---|---|
| 1 | Estructura inicial | `git add README.md .gitignore requirements.txt modelos/.gitkeep` → `git commit -m "Estructura inicial del proyecto"` |
| 2 | Dataset y su fuente | `git add data/ scripts/` → `git commit -m "Agregar dataset Breast Cancer Wisconsin (UCI) y script de exportación"` |
| 3 | Limpieza de datos | `git add ml/__init__.py ml/datos.py` → `git commit -m "Carga y limpieza de datos"` |
| 4 | Entrenamiento y métricas | `git add ml/entrenamiento.py` → `git commit -m "Entrenamiento de modelos y cálculo de métricas"` |
| 5 | API | `git add api/` → `git commit -m "API REST con FastAPI: entrenar, métricas y predecir"` |
| 6 | Pruebas | `git add tests/` → `git commit -m "Pruebas automáticas del API"` |
| 7 | Interfaz gráfica | `git add frontend/` → `git commit -m "Interfaz gráfica en Streamlit"` |
| 8 | Ajustes finales | `git add -A` → `git commit -m "Documentación y ajustes finales"` |

Después de cada commit: `git push -u origin main`.

Si se suman compañeros al grupo, cada uno debe hacer sus propios commits desde su cuenta (por ejemplo, uno sube la interfaz y otro las pruebas).
