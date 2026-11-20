"""Exporta el dataset Breast Cancer Wisconsin (Diagnostic) a CSV.

El dataset original es de la UCI Machine Learning Repository
(Wolberg, Mangasarian, Street y Street, 1995). scikit-learn incluye una
copia exacta del archivo wdbc.data, por eso lo usamos para generar el CSV
sin depender de internet.
"""
from pathlib import Path

from sklearn.datasets import load_breast_cancer

RUTA = Path(__file__).resolve().parents[1] / "data" / "breast_cancer_wisconsin.csv"


def main() -> None:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # En scikit-learn: 0 = maligno, 1 = benigno. Volvemos a la notación original de la UCI (M/B).
    df["diagnostico"] = df.pop("target").map({0: "M", 1: "B"})
    df.insert(0, "id", range(1, len(df) + 1))
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA, index=False)
    print(f"Dataset exportado: {RUTA} ({len(df)} filas, {df.shape[1]} columnas)")


if __name__ == "__main__":
    main()
