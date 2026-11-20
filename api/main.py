"""API REST del modelo de predicción de cáncer de mama.

Ejecutar:  uvicorn api.main:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.datos import cargar_datos, columnas_caracteristicas, limpiar_datos, nombre_es
from ml.entrenamiento import ALGORITMOS, cargar_modelo, entrenar, leer_historial, predecir

app = FastAPI(
    title="API - Predicción de cáncer de mama",
    description="Entrena, evalúa y usa modelos de Machine Learning sobre el dataset "
                "Breast Cancer Wisconsin (Diagnostic) de la UCI.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SolicitudEntrenamiento(BaseModel):
    algoritmo: Literal["regresion_logistica", "random_forest", "svm", "knn"] = "random_forest"
    parametros: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Proporción de datos para prueba")
    semilla: int = Field(42, ge=0, description="Semilla para que los resultados sean repetibles")
    escalar: bool = Field(True, description="Estandarizar las características")
    folds_cv: int = Field(5, ge=2, le=10, description="Particiones de validación cruzada")


class SolicitudPrediccion(BaseModel):
    caracteristicas: dict[str, float] = Field(..., description="Las 30 medidas del caso")
    umbral: float = Field(0.5, gt=0, lt=1, description="Probabilidad mínima para decir 'maligno'")


@app.get("/salud")
def salud() -> dict[str, Any]:
    paquete = cargar_modelo()
    return {"estado": "ok", "modelo_entrenado": paquete is not None,
            "modelo_id": paquete["resultado"]["id"] if paquete else None}


@app.get("/dataset/resumen")
def resumen_dataset() -> dict[str, Any]:
    df, reporte = limpiar_datos(cargar_datos())
    columnas = columnas_caracteristicas(df)
    estadisticas = df[columnas].describe().T.round(4)
    return {
        "fuente": "UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic). "
                  "https://doi.org/10.24432/C5DW2B",
        "filas": int(len(df)),
        "caracteristicas": [{"columna": c, "nombre": nombre_es(c)} for c in columnas],
        "limpieza": reporte,
        "estadisticas": estadisticas.reset_index(names="columna").to_dict(orient="records"),
    }


@app.get("/dataset/ejemplo")
def ejemplo(tipo: Literal["M", "B", "cualquiera"] = "cualquiera",
            semilla: int | None = Query(None)) -> dict[str, Any]:
    """Devuelve un caso real del dataset para probar la predicción."""
    df, _ = limpiar_datos(cargar_datos())
    if tipo != "cualquiera":
        df = df[df["diagnostico"] == tipo]
    fila = df.sample(1, random_state=semilla).iloc[0]
    return {"id": int(fila["id"]), "diagnostico_real": fila["diagnostico"],
            "caracteristicas": {c: float(fila[c]) for c in columnas_caracteristicas(df)}}


@app.get("/algoritmos")
def algoritmos() -> dict[str, Any]:
    return ALGORITMOS


@app.post("/entrenar")
def entrenar_modelo(solicitud: SolicitudEntrenamiento) -> dict[str, Any]:
    try:
        return entrenar(**solicitud.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/metricas")
def metricas() -> dict[str, Any]:
    paquete = cargar_modelo()
    if paquete is None:
        raise HTTPException(status_code=404, detail="No hay un modelo entrenado todavía.")
    return paquete["resultado"]


@app.get("/historial")
def historial() -> list[dict[str, Any]]:
    return leer_historial()


@app.post("/predecir")
def predecir_caso(solicitud: SolicitudPrediccion) -> dict[str, Any]:
    try:
        return predecir(solicitud.caracteristicas, solicitud.umbral)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
