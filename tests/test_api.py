from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)


def test_salud():
    assert cliente.get("/salud").json()["estado"] == "ok"


def test_resumen_dataset():
    datos = cliente.get("/dataset/resumen").json()
    assert datos["filas"] == 569
    assert len(datos["caracteristicas"]) == 30


def test_entrenar_y_predecir():
    r = cliente.post("/entrenar", json={"algoritmo": "regresion_logistica"})
    assert r.status_code == 200
    assert r.json()["metricas"]["exactitud"] > 0.9

    ejemplo = cliente.get("/dataset/ejemplo", params={"tipo": "M", "semilla": 1}).json()
    p = cliente.post("/predecir", json={"caracteristicas": ejemplo["caracteristicas"]})
    assert p.status_code == 200
    assert p.json()["diagnostico"] in ("M", "B")
    assert len(p.json()["explicacion"]) == 8


def test_parametro_invalido():
    r = cliente.post("/entrenar", json={"algoritmo": "knn", "parametros": {"n_neighbors": 999}})
    assert r.status_code == 422


def test_caso_incompleto():
    r = cliente.post("/predecir", json={"caracteristicas": {"mean radius": 10}})
    assert r.status_code in (409, 422)
