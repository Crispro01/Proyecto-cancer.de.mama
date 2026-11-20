"""Entrenamiento, evaluación, guardado y predicción del modelo."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.datos import RAIZ, cargar_datos, limpiar_datos, nombre_es, separar_x_y

CARPETA_MODELOS = RAIZ / "modelos"
RUTA_MODELO = CARPETA_MODELOS / "modelo_actual.joblib"
RUTA_HISTORIAL = CARPETA_MODELOS / "historial.json"

# Algoritmos disponibles y los parámetros que el usuario puede configurar desde la interfaz
ALGORITMOS: dict[str, dict[str, Any]] = {
    "regresion_logistica": {
        "nombre": "Regresión Logística",
        "parametros": {
            "C": {"tipo": "float", "defecto": 1.0, "min": 0.01, "max": 100.0,
                  "ayuda": "Inverso de la regularización. Más alto = modelo menos restringido."},
            "max_iter": {"tipo": "int", "defecto": 1000, "min": 100, "max": 5000,
                         "ayuda": "Iteraciones máximas del optimizador."},
        },
    },
    "random_forest": {
        "nombre": "Random Forest",
        "parametros": {
            "n_estimators": {"tipo": "int", "defecto": 200, "min": 10, "max": 1000,
                             "ayuda": "Número de árboles del bosque."},
            "max_depth": {"tipo": "int", "defecto": 0, "min": 0, "max": 50,
                          "ayuda": "Profundidad máxima de cada árbol (0 = sin límite)."},
            "min_samples_split": {"tipo": "int", "defecto": 2, "min": 2, "max": 20,
                                  "ayuda": "Muestras mínimas para dividir un nodo."},
        },
    },
    "svm": {
        "nombre": "Máquina de Vectores de Soporte (SVM)",
        "parametros": {
            "C": {"tipo": "float", "defecto": 1.0, "min": 0.01, "max": 100.0,
                  "ayuda": "Penalización por errores de clasificación."},
            "kernel": {"tipo": "opcion", "defecto": "rbf", "opciones": ["rbf", "linear", "poly"],
                       "ayuda": "Función núcleo usada para separar las clases."},
        },
    },
    "knn": {
        "nombre": "K Vecinos Más Cercanos (KNN)",
        "parametros": {
            "n_neighbors": {"tipo": "int", "defecto": 5, "min": 1, "max": 50,
                            "ayuda": "Cantidad de vecinos que votan."},
            "weights": {"tipo": "opcion", "defecto": "uniform", "opciones": ["uniform", "distance"],
                        "ayuda": "uniform = todos votan igual; distance = vecinos cercanos pesan más."},
        },
    },
}


def _crear_estimador(algoritmo: str, p: dict[str, Any], semilla: int):
    if algoritmo == "regresion_logistica":
        return LogisticRegression(C=float(p["C"]), max_iter=int(p["max_iter"]), random_state=semilla)
    if algoritmo == "random_forest":
        profundidad = int(p["max_depth"]) or None
        return RandomForestClassifier(
            n_estimators=int(p["n_estimators"]), max_depth=profundidad,
            min_samples_split=int(p["min_samples_split"]), random_state=semilla, n_jobs=-1,
        )
    if algoritmo == "svm":
        # CalibratedClassifierCV convierte la salida del SVM en probabilidades calibradas
        svm = SVC(C=float(p["C"]), kernel=p["kernel"], random_state=semilla)
        return CalibratedClassifierCV(svm, cv=5, ensemble=False)
    if algoritmo == "knn":
        return KNeighborsClassifier(n_neighbors=int(p["n_neighbors"]), weights=p["weights"])
    raise ValueError(f"Algoritmo no soportado: {algoritmo}")


def completar_parametros(algoritmo: str, parametros: dict[str, Any] | None) -> dict[str, Any]:
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"Algoritmo no soportado: {algoritmo}. Opciones: {list(ALGORITMOS)}")
    definicion = ALGORITMOS[algoritmo]["parametros"]
    final = {k: v["defecto"] for k, v in definicion.items()}
    for k, v in (parametros or {}).items():
        if k not in definicion:
            raise ValueError(f"Parámetro desconocido para {algoritmo}: {k}")
        d = definicion[k]
        if d["tipo"] == "opcion" and v not in d["opciones"]:
            raise ValueError(f"{k} debe ser uno de {d['opciones']}")
        if d["tipo"] in ("int", "float") and not (d["min"] <= float(v) <= d["max"]):
            raise ValueError(f"{k} debe estar entre {d['min']} y {d['max']}")
        final[k] = v
    return final


def entrenar(
    algoritmo: str = "random_forest",
    parametros: dict[str, Any] | None = None,
    test_size: float = 0.2,
    semilla: int = 42,
    escalar: bool = True,
    folds_cv: int = 5,
) -> dict[str, Any]:
    """Entrena un modelo, calcula sus métricas, lo guarda y devuelve el resultado."""
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size debe estar entre 0.1 y 0.5")
    if not 2 <= folds_cv <= 10:
        raise ValueError("folds_cv debe estar entre 2 y 10")
    parametros = completar_parametros(algoritmo, parametros)

    df, reporte_limpieza = limpiar_datos(cargar_datos())
    x, y = separar_x_y(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=semilla, stratify=y
    )

    pasos = [("escalador", StandardScaler())] if escalar else []
    pasos.append(("modelo", _crear_estimador(algoritmo, parametros, semilla)))
    pipeline = Pipeline(pasos)

    inicio = time.perf_counter()
    cv = StratifiedKFold(n_splits=folds_cv, shuffle=True, random_state=semilla)
    puntajes_cv = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1")
    pipeline.fit(x_train, y_train)
    segundos = time.perf_counter() - inicio

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    importancia = permutation_importance(
        pipeline, x_test, y_test, n_repeats=10, random_state=semilla, scoring="f1"
    )
    orden = np.argsort(importancia.importances_mean)[::-1][:10]
    top = [
        {"caracteristica": x.columns[i], "nombre": nombre_es(x.columns[i]),
         "importancia": round(float(importancia.importances_mean[i]), 4)}
        for i in orden
    ]

    metricas = {
        "exactitud": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "sensibilidad": round(float(recall_score(y_test, y_pred)), 4),
        "especificidad": round(float(tn / (tn + fp)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1_validacion_cruzada_media": round(float(puntajes_cv.mean()), 4),
        "f1_validacion_cruzada_desv": round(float(puntajes_cv.std()), 4),
    }

    resultado = {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "algoritmo": algoritmo,
        "nombre_algoritmo": ALGORITMOS[algoritmo]["nombre"],
        "parametros": parametros,
        "configuracion": {"test_size": test_size, "semilla": semilla, "escalar": escalar,
                          "folds_cv": folds_cv},
        "datos": {"n_entrenamiento": int(len(x_train)), "n_prueba": int(len(x_test)),
                  "limpieza": reporte_limpieza},
        "metricas": metricas,
        "matriz_confusion": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
        "curva_roc": {"fpr": [round(float(v), 4) for v in fpr],
                      "tpr": [round(float(v), 4) for v in tpr]},
        "validacion_cruzada": [round(float(v), 4) for v in puntajes_cv],
        "importancia_caracteristicas": top,
        "segundos_entrenamiento": round(segundos, 2),
    }

    CARPETA_MODELOS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "columnas": list(x.columns), "resultado": resultado,
                 "medianas": {c: float(v) for c, v in x_train.median().items()}},
                RUTA_MODELO)
    _guardar_en_historial(resultado)
    return resultado


def _guardar_en_historial(resultado: dict[str, Any]) -> None:
    historial = leer_historial()
    resumen = {k: resultado[k] for k in ("id", "fecha", "nombre_algoritmo", "parametros",
                                         "configuracion", "metricas")}
    historial.append(resumen)
    RUTA_HISTORIAL.write_text(json.dumps(historial[-50:], indent=2, ensure_ascii=False),
                              encoding="utf-8")


def leer_historial() -> list[dict[str, Any]]:
    if not RUTA_HISTORIAL.exists():
        return []
    try:
        return json.loads(RUTA_HISTORIAL.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def cargar_modelo() -> dict[str, Any] | None:
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)


def predecir(caso: dict[str, float], umbral: float = 0.5) -> dict[str, Any]:
    paquete = cargar_modelo()
    if paquete is None:
        raise RuntimeError("Todavía no hay un modelo entrenado. Entrena uno primero.")
    columnas = paquete["columnas"]
    faltantes = [c for c in columnas if c not in caso]
    if faltantes:
        raise ValueError(f"Faltan {len(faltantes)} características, por ejemplo: {faltantes[:3]}")
    negativos = [c for c in columnas if float(caso[c]) < 0]
    if negativos:
        raise ValueError(f"Las medidas no pueden ser negativas: {negativos[:3]}")
    fila = pd.DataFrame([[float(caso[c]) for c in columnas]], columns=columnas)
    prob_maligno = float(paquete["pipeline"].predict_proba(fila)[0, 1])
    es_maligno = prob_maligno >= umbral
    explicacion = explicar_caso(paquete, fila, prob_maligno)
    return {
        "diagnostico": "M" if es_maligno else "B",
        "etiqueta": "Maligno" if es_maligno else "Benigno",
        "probabilidad_maligno": round(prob_maligno, 4),
        "probabilidad_benigno": round(1 - prob_maligno, 4),
        "umbral": umbral,
        "modelo": paquete["resultado"]["nombre_algoritmo"],
        "modelo_id": paquete["resultado"]["id"],
        "confianza_baja": abs(prob_maligno - umbral) < 0.15,
        "explicacion": explicacion,
        "aviso": "Resultado con fines educativos. No reemplaza el diagnóstico de un médico.",
    }


def explicar_caso(paquete: dict[str, Any], fila: pd.DataFrame, prob: float,
                  top: int = 8) -> list[dict[str, Any]]:
    """Explica una predicción individual (idea parecida a Grad-CAM, pero para datos numéricos).

    Para cada medida, la reemplazamos por el valor típico (la mediana del entrenamiento) y vemos
    cuánto cambia la probabilidad de malignidad. Si al "normalizar" una medida la probabilidad
    baja, esa medida estaba empujando el caso hacia maligno (contribución positiva).
    """
    medianas = paquete.get("medianas")
    if medianas is None:  # modelos guardados con una versión anterior
        df, _ = limpiar_datos(cargar_datos())
        medianas = {c: float(df[c].median()) for c in paquete["columnas"]}
    variantes = pd.concat([fila] * len(paquete["columnas"]), ignore_index=True)
    for i, c in enumerate(paquete["columnas"]):
        variantes.loc[i, c] = medianas[c]
    probs = paquete["pipeline"].predict_proba(variantes)[:, 1]

    # Trabajamos en log-odds para que la explicación funcione aunque la probabilidad esté
    # muy cerca de 0% o 100% (ahí las diferencias en probabilidad casi no se notan).
    def logit(v):
        v = np.clip(v, 1e-6, 1 - 1e-6)
        return np.log(v / (1 - v))

    deltas = logit(prob) - logit(probs)
    total = float(np.abs(deltas).sum()) or 1.0
    aportes = [
        {"caracteristica": c, "nombre": nombre_es(c), "valor": round(float(fila.iloc[0][c]), 4),
         "valor_tipico": round(medianas[c], 4),
         "contribucion": round(float(deltas[i]), 4),
         "peso_porcentaje": round(100 * abs(float(deltas[i])) / total, 1),
         "direccion": "maligno" if deltas[i] > 0 else "benigno"}
        for i, c in enumerate(paquete["columnas"])
    ]
    aportes.sort(key=lambda a: abs(a["contribucion"]), reverse=True)
    return aportes[:top]
