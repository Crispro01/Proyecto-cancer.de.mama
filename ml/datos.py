"""Carga y limpieza del dataset Breast Cancer Wisconsin (Diagnostic)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATASET = RAIZ / "data" / "breast_cancer_wisconsin.csv"
COLUMNA_OBJETIVO = "diagnostico"
CLASE_POSITIVA = "M"  # Maligno: es la clase que más nos importa detectar

# Traducción de las 10 características base (cada una tiene media, error estándar y peor valor)
CARACTERISTICAS_BASE = {
    "radius": "radio",
    "texture": "textura",
    "perimeter": "perímetro",
    "area": "área",
    "smoothness": "suavidad",
    "compactness": "compacidad",
    "concavity": "concavidad",
    "concave points": "puntos cóncavos",
    "symmetry": "simetría",
    "fractal dimension": "dimensión fractal",
}
PREFIJOS = {"mean": "media", "error": "error estándar", "worst": "peor valor"}


def nombre_es(columna: str) -> str:
    """Convierte 'mean radius' -> 'radio (media)'."""
    for base, es in CARACTERISTICAS_BASE.items():
        if columna == f"mean {base}":
            return f"{es} (media)"
        if columna == f"{base} error":
            return f"{es} (error estándar)"
        if columna == f"worst {base}":
            return f"{es} (peor valor)"
    return columna


def cargar_datos(ruta: Path = RUTA_DATASET) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {ruta}. Ejecuta: python scripts/exportar_dataset.py"
        )
    return pd.read_csv(ruta)


def columnas_caracteristicas(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("id", COLUMNA_OBJETIVO)]


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpia el dataset y devuelve un reporte de lo que se hizo.

    Pasos:
    1. Quitar filas duplicadas.
    2. Quitar filas sin diagnóstico o con un diagnóstico distinto de M/B.
    3. Convertir las características a número (lo que no se pueda queda como NaN).
    4. Valores negativos -> NaN (ninguna medida física puede ser negativa).
    5. Imputar NaN con la mediana de cada columna.
    6. Contar valores atípicos (regla IQR). Se reportan, no se eliminan, porque en
       medicina un valor extremo suele ser justamente un tumor maligno.
    """
    reporte: dict = {"filas_iniciales": int(len(df))}
    df = df.copy()

    antes = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])
    reporte["duplicados_eliminados"] = int(antes - len(df))

    antes = len(df)
    df[COLUMNA_OBJETIVO] = df[COLUMNA_OBJETIVO].astype(str).str.strip().str.upper()
    df = df[df[COLUMNA_OBJETIVO].isin(["M", "B"])]
    reporte["filas_diagnostico_invalido"] = int(antes - len(df))

    columnas = columnas_caracteristicas(df)
    df[columnas] = df[columnas].apply(pd.to_numeric, errors="coerce")

    negativos = int((df[columnas] < 0).sum().sum())
    df[columnas] = df[columnas].mask(df[columnas] < 0)
    reporte["valores_negativos_corregidos"] = negativos

    nulos = int(df[columnas].isna().sum().sum())
    df[columnas] = df[columnas].fillna(df[columnas].median())
    reporte["valores_nulos_imputados"] = nulos

    q1, q3 = df[columnas].quantile(0.25), df[columnas].quantile(0.75)
    iqr = q3 - q1
    atipicos = ((df[columnas] < q1 - 1.5 * iqr) | (df[columnas] > q3 + 1.5 * iqr)).sum()
    reporte["valores_atipicos_detectados"] = int(atipicos.sum())
    reporte["atipicos_por_columna"] = {k: int(v) for k, v in atipicos.items() if v > 0}

    reporte["filas_finales"] = int(len(df))
    reporte["distribucion_clases"] = {
        k: int(v) for k, v in df[COLUMNA_OBJETIVO].value_counts().items()
    }
    return df.reset_index(drop=True), reporte


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = df[columnas_caracteristicas(df)]
    y = (df[COLUMNA_OBJETIVO] == CLASE_POSITIVA).astype(int).to_numpy()  # 1 = maligno
    return x, y
