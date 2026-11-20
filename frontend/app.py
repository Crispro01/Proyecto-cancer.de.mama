"""Interfaz gráfica (frontend) en Streamlit. Se comunica con el API de FastAPI.

Ejecutar:  streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
COLOR_MALIGNO = "#BE185D"   # rosa fuerte
COLOR_BENIGNO = "#0F766E"   # verde azulado
NOMBRE_APP = "OncoMama IA"

st.set_page_config(page_title=NOMBRE_APP, page_icon="🩷", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 3.2rem;}
[data-testid="stSidebar"] {background: #FCE7EF;}
.om-header {background: #FFFFFF; border: 1px solid #F5C6D6; border-left: 0; border-radius: 18px;
  padding: 18px 26px; display: flex; align-items: center; gap: 20px; margin-bottom: 18px;
  box-shadow: 0 2px 10px rgba(190, 24, 93, 0.08);}
.om-logo {width: 64px; height: 64px; border-radius: 16px; background: #FCE7EF;
  display: flex; align-items: center; justify-content: center; flex: none;}
.om-title {font-size: 30px; font-weight: 800; color: #9D174D; line-height: 1.1; margin: 0;}
.om-title span {color: #3B1D2B; font-weight: 600;}
.om-sub {font-size: 14px; color: #6B4A5A; margin: 4px 0 0 0;}
.om-status {margin-left: auto; text-align: right; white-space: nowrap; font-size: 13px; color: #6B4A5A; line-height: 1.7;}
.om-dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;}
.om-card {background: #FFFFFF; border: 2px solid #F3D9E3; border-radius: 16px; padding: 18px 20px;}
.om-card.principal-m {border-color: #BE185D; background: #FDF2F7;}
.om-card.principal-b {border-color: #0F766E; background: #EEF8F6;}
.om-card .lbl {font-size: 15px; font-weight: 700; margin: 0;}
.om-card .pct {font-size: 40px; font-weight: 800; margin: 6px 0 0 0; line-height: 1;}
.om-card .tag {font-size: 12px; font-weight: 700; letter-spacing: 1px; float: right;}
</style>
""", unsafe_allow_html=True)

LOGO_SVG = (
    '<svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#BE185D" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 3c-2.2 0-3.5 1.6-3.5 3.6 0 2.4 2 5 3.5 7.4 1.5-2.4 3.5-5 3.5-7.4C15.5 4.6 14.2 3 12 3z"/>'
    '<path d="M10.2 11.3 6 21l3.2-1.2L11 22l2-5.2"/><path d="M13.8 11.3 18 21l-3.2-1.2L13 22"/></svg>'
)


def encabezado(salud: dict | None) -> None:
    conectado = salud is not None
    modelo = salud.get("modelo_id") if salud else None
    color_api = "#16A34A" if conectado else "#DC2626"
    color_mod = "#16A34A" if modelo else "#D97706"
    st.markdown(f"""
<div class="om-header">
  <div class="om-logo">{LOGO_SVG}</div>
  <div>
    <p class="om-title">{NOMBRE_APP} <span>| Predicción de cáncer de mama</span></p>
    <p class="om-sub">Sistema de apoyo al diagnóstico citológico con Machine Learning ·
    Breast Cancer Wisconsin (UCI)</p>
  </div>
  <div class="om-status">
    <div><span class="om-dot" style="background:{color_api}"></span>
    {"Conectado al API REST" if conectado else "API desconectado"}</div>
    <div><span class="om-dot" style="background:{color_mod}"></span>
    {"Modelo activo: " + modelo if modelo else "Sin modelo entrenado"}</div>
    <div>cristina.proano@cenestur.edu.ec</div>
  </div>
</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- utilidades
def api(metodo: str, ruta: str, **kwargs):
    try:
        respuesta = requests.request(metodo, f"{API_URL}{ruta}", timeout=120, **kwargs)
    except requests.ConnectionError:
        st.error(f"No me pude conectar al API en {API_URL}. ¿Está corriendo "
                 "`uvicorn api.main:app --reload`?")
        st.stop()
    if respuesta.status_code >= 400:
        detalle = respuesta.json().get("detail", respuesta.text)
        return None, detalle
    return respuesta.json(), None


@st.cache_data(ttl=600)
def resumen_dataset():
    datos, _ = api("GET", "/dataset/resumen")
    return datos


@st.cache_data(ttl=600)
def algoritmos():
    datos, _ = api("GET", "/algoritmos")
    return datos


def tarjetas_metricas(m: dict) -> None:
    columnas = st.columns(6)
    etiquetas = [("Exactitud", "exactitud"), ("Precisión", "precision"),
                 ("Sensibilidad", "sensibilidad"), ("Especificidad", "especificidad"),
                 ("F1", "f1"), ("ROC AUC", "roc_auc")]
    for col, (texto, clave) in zip(columnas, etiquetas):
        col.metric(texto, f"{m[clave] * 100:.1f}%")


def graficos_resultado(r: dict) -> None:
    tarjetas_metricas(r["metricas"])
    st.caption(
        f"Validación cruzada ({r['configuracion']['folds_cv']} folds) sobre entrenamiento: "
        f"F1 = {r['metricas']['f1_validacion_cruzada_media'] * 100:.1f}% "
        f"± {r['metricas']['f1_validacion_cruzada_desv'] * 100:.1f}%  ·  "
        f"Entrenado en {r['segundos_entrenamiento']} s con {r['datos']['n_entrenamiento']} casos; "
        f"evaluado con {r['datos']['n_prueba']} casos que el modelo nunca vio."
    )

    c1, c2 = st.columns(2)
    mc = r["matriz_confusion"]
    with c1:
        st.markdown("**Matriz de confusión**")
        z = [[mc["vn"], mc["fp"]], [mc["fn"], mc["vp"]]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicho benigno", "Predicho maligno"], y=["Real benigno", "Real maligno"],
            text=z, texttemplate="%{text}", textfont={"size": 22},
            colorscale=[[0, "#FDF2F7"], [1, "#BE185D"]], showscale=False))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Falsos negativos (malignos que se escaparon): **{mc['fn']}**. "
                   "En un tamizaje médico este es el error más costoso.")
    with c2:
        st.markdown(f"**Curva ROC** (AUC = {r['metricas']['roc_auc']:.3f})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["curva_roc"]["fpr"], y=r["curva_roc"]["tpr"],
                                 mode="lines", line=dict(color=COLOR_BENIGNO, width=3),
                                 name="Modelo"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color="#94A3B8", dash="dash"), name="Azar"))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                          xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Características más importantes** (importancia por permutación)")
    imp = pd.DataFrame(r["importancia_caracteristicas"]).sort_values("importancia")
    fig = px.bar(imp, x="importancia", y="nombre", orientation="h",
                 color_discrete_sequence=[COLOR_BENIGNO])
    fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10),
                      xaxis_title="Caída del F1 al desordenar la característica", yaxis_title="")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- navegación
st.sidebar.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>{LOGO_SVG}<span style='font-size:26px;font-weight:800;color:#9D174D'>{NOMBRE_APP}</span></div>", unsafe_allow_html=True)
pagina = st.sidebar.radio("Ir a", ["Inicio", "Datos", "Entrenar modelo", "Métricas",
                                   "Nuevo caso", "Historial"])
salud, _ = api("GET", "/salud")
encabezado(salud)
if salud and salud["modelo_entrenado"]:
    st.sidebar.success(f"Modelo activo: {salud['modelo_id']}")
else:
    st.sidebar.warning("Aún no hay modelo entrenado")
st.sidebar.caption(f"API: {API_URL}")
st.sidebar.caption("Proyecto educativo. No es una herramienta de diagnóstico médico.")


# ---------------------------------------------------------------- páginas
if pagina == "Inicio":
    st.title("Bienvenida a OncoMama IA")
    st.write(
        "Esta aplicación usa medidas de los núcleos celulares obtenidas de imágenes de biopsias "
        "por aspiración con aguja fina (FNA) para estimar si un tumor es **benigno** o "
        "**maligno**. Puedes entrenar distintos modelos, cambiar sus parámetros, revisar sus "
        "métricas y probar un caso nuevo."
    )
    c1, c2, c3 = st.columns(3)
    for col, (titulo, texto) in zip([c1, c2, c3], [
            ("1. Entrenar", "Elige el algoritmo y sus parámetros."),
            ("2. Evaluar", "Revisa exactitud, sensibilidad, matriz de confusión y curva ROC."),
            ("3. Predecir", "Ingresa un caso nuevo y obtén la probabilidad de malignidad.")]):
        col.markdown(f"<div class='om-card' style='min-height:120px'><p class='lbl' "
                     f"style='color:#9D174D'>{titulo}</p><p style='margin:8px 0 0 0'>{texto}</p></div>",
                     unsafe_allow_html=True)
    st.markdown(
        "**Fuente de los datos:** Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1995). "
        "*Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5DW2B"
    )

elif pagina == "Datos":
    st.title("Exploración y limpieza de datos")
    datos = resumen_dataset()
    lim = datos["limpieza"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos", datos["filas"])
    c2.metric("Características", len(datos["caracteristicas"]))
    c3.metric("Malignos", lim["distribucion_clases"].get("M", 0))
    c4.metric("Benignos", lim["distribucion_clases"].get("B", 0))

    st.subheader("Reporte de limpieza")
    st.table(pd.DataFrame({
        "Paso": ["Filas iniciales", "Duplicados eliminados", "Diagnósticos inválidos eliminados",
                 "Valores negativos corregidos", "Valores nulos imputados (mediana)",
                 "Valores atípicos detectados (IQR, se conservan)", "Filas finales"],
        "Resultado": [lim["filas_iniciales"], lim["duplicados_eliminados"],
                      lim["filas_diagnostico_invalido"], lim["valores_negativos_corregidos"],
                      lim["valores_nulos_imputados"], lim["valores_atipicos_detectados"],
                      lim["filas_finales"]],
    }))
    st.caption("Los atípicos no se eliminan: en este problema un valor extremo suele indicar un "
               "tumor maligno, así que quitarlos haría que el modelo aprenda peor.")

    st.subheader("Distribución de clases")
    dist = pd.DataFrame({"Diagnóstico": ["Benigno", "Maligno"],
                         "Casos": [lim["distribucion_clases"].get("B", 0),
                                   lim["distribucion_clases"].get("M", 0)]})
    fig = px.bar(dist, x="Diagnóstico", y="Casos", color="Diagnóstico",
                 color_discrete_map={"Benigno": COLOR_BENIGNO, "Maligno": COLOR_MALIGNO})
    fig.update_layout(height=320, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Estadísticas descriptivas")
    est = pd.DataFrame(datos["estadisticas"])
    nombres = {c["columna"]: c["nombre"] for c in datos["caracteristicas"]}
    est.insert(0, "característica", est["columna"].map(nombres))
    st.dataframe(est.drop(columns=["columna", "count"]), width="stretch", hide_index=True)

elif pagina == "Entrenar modelo":
    st.title("Configurar y entrenar el modelo")
    algos = algoritmos()
    clave = st.selectbox("Algoritmo", list(algos), format_func=lambda k: algos[k]["nombre"])

    st.subheader("Hiperparámetros")
    parametros = {}
    cols = st.columns(len(algos[clave]["parametros"]))
    for col, (nombre, d) in zip(cols, algos[clave]["parametros"].items()):
        with col:
            if d["tipo"] == "int":
                parametros[nombre] = st.number_input(nombre, int(d["min"]), int(d["max"]),
                                                     int(d["defecto"]), help=d["ayuda"])
            elif d["tipo"] == "float":
                parametros[nombre] = st.number_input(nombre, float(d["min"]), float(d["max"]),
                                                     float(d["defecto"]), help=d["ayuda"])
            else:
                parametros[nombre] = st.selectbox(nombre, d["opciones"],
                                                  index=d["opciones"].index(d["defecto"]),
                                                  help=d["ayuda"])

    st.subheader("Configuración del entrenamiento")
    c1, c2, c3, c4 = st.columns(4)
    test_size = c1.slider("Datos para prueba", 0.1, 0.5, 0.2, 0.05)
    folds = c2.slider("Folds de validación cruzada", 2, 10, 5)
    semilla = c3.number_input("Semilla aleatoria", 0, 9999, 42)
    escalar = c4.checkbox("Estandarizar datos", value=True)

    if st.button("🚀 Entrenar modelo", type="primary"):
        with st.spinner("Entrenando y evaluando..."):
            resultado, error = api("POST", "/entrenar", json={
                "algoritmo": clave, "parametros": parametros, "test_size": test_size,
                "semilla": int(semilla), "escalar": escalar, "folds_cv": folds})
        if error:
            st.error(error)
        else:
            st.success(f"Modelo {resultado['nombre_algoritmo']} entrenado y guardado "
                       f"(id {resultado['id']}).")
            graficos_resultado(resultado)

elif pagina == "Métricas":
    st.title("Métricas del modelo activo")
    resultado, error = api("GET", "/metricas")
    if error:
        st.info(error + " Ve a **Entrenar modelo**.")
    else:
        st.write(f"**{resultado['nombre_algoritmo']}** · parámetros: `{resultado['parametros']}` "
                 f"· entrenado el {resultado['fecha'].replace('T', ' ')}")
        graficos_resultado(resultado)
        with st.expander("¿Qué significa cada métrica?"):
            st.markdown(
                "- **Exactitud:** porcentaje total de aciertos.\n"
                "- **Precisión:** de los casos que el modelo dijo *maligno*, cuántos lo eran.\n"
                "- **Sensibilidad (recall):** de los malignos reales, cuántos detectó. "
                "Es la métrica más importante en salud.\n"
                "- **Especificidad:** de los benignos reales, cuántos reconoció como benignos.\n"
                "- **F1:** equilibrio entre precisión y sensibilidad.\n"
                "- **ROC AUC:** qué tan bien separa las dos clases (1 = perfecto, 0.5 = azar).")

elif pagina == "Nuevo caso":
    st.title("Predecir un caso nuevo")
    datos = resumen_dataset()
    nombres = {c["columna"]: c["nombre"] for c in datos["caracteristicas"]}
    medianas = {e["columna"]: e["50%"] for e in datos["estadisticas"]}

    if "real" not in st.session_state:
        st.session_state.real = None
    for k, v in medianas.items():
        st.session_state.setdefault(f"in_{k}", float(v))

    st.write("Ingresa las 30 medidas o carga un caso real del dataset para probar.")
    c1, c2, c3 = st.columns(3)
    for col, (texto, tipo) in zip([c1, c2, c3], [("Cargar ejemplo maligno", "M"),
                                                 ("Cargar ejemplo benigno", "B"),
                                                 ("Valores medianos", None)]):
        if col.button(texto, width="stretch"):
            if tipo:
                ej, _ = api("GET", "/dataset/ejemplo", params={"tipo": tipo})
                valores = ej["caracteristicas"]
                st.session_state.real = f"Caso #{ej['id']} — diagnóstico real: " + \
                    ("Maligno" if ej["diagnostico_real"] == "M" else "Benigno")
            else:
                valores = medianas
                st.session_state.real = None
            for k, v in valores.items():
                st.session_state[f"in_{k}"] = float(v)
    if st.session_state.real:
        st.caption(st.session_state.real)

    columnas = list(nombres)
    grupos = {"Media": columnas[0:10], "Error estándar": columnas[10:20],
              "Peor valor": columnas[20:30]}
    caso = {}
    for pestana, cols_grupo in zip(st.tabs(list(grupos)), grupos.values()):
        with pestana:
            filas = st.columns(2)
            for i, c in enumerate(cols_grupo):
                with filas[i % 2]:
                    caso[c] = st.number_input(nombres[c], min_value=0.0, format="%.4f",
                                              key=f"in_{c}")

    umbral = st.slider("Umbral de decisión (probabilidad mínima para clasificar como maligno)",
                       0.05, 0.95, 0.5, 0.05,
                       help="Bajarlo aumenta la sensibilidad: se escapan menos malignos, "
                            "a cambio de más falsas alarmas.")

    if st.button("🔍 Predecir", type="primary"):
        pred, error = api("POST", "/predecir", json={"caracteristicas": caso, "umbral": umbral})
        if error:
            st.error(error)
        else:
            es_m = pred["diagnostico"] == "M"
            color = COLOR_MALIGNO if es_m else COLOR_BENIGNO
            st.markdown(f"<h2 style='color:{color};margin-bottom:0'>Resultado: {pred['etiqueta']}</h2>",
                        unsafe_allow_html=True)
            if pred["confianza_baja"]:
                st.warning(f"⚠️ Confianza baja: la probabilidad ({pred['probabilidad_maligno']*100:.1f}%) "
                           f"está cerca del umbral ({umbral*100:.0f}%). Se recomienda revisión manual.")

            c1, c2, c3 = st.columns([1, 1, 1.3])
            with c1:
                st.markdown(f"""<div class="om-card {'principal-b' if not es_m else ''}">
<span class="tag" style="color:{COLOR_BENIGNO}">{'PRINCIPAL' if not es_m else ''}</span>
<p class="lbl" style="color:{COLOR_BENIGNO}">● Benigno</p>
<p class="pct" style="color:{COLOR_BENIGNO}">{pred['probabilidad_benigno']*100:.1f}%</p></div>""",
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="om-card {'principal-m' if es_m else ''}">
<span class="tag" style="color:{COLOR_MALIGNO}">{'PRINCIPAL' if es_m else ''}</span>
<p class="lbl" style="color:{COLOR_MALIGNO}">● Maligno</p>
<p class="pct" style="color:{COLOR_MALIGNO}">{pred['probabilidad_maligno']*100:.1f}%</p></div>""",
                            unsafe_allow_html=True)
            with c3:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=pred["probabilidad_maligno"] * 100,
                    number={"suffix": "%", "font": {"color": color}},
                    title={"text": "Probabilidad de malignidad", "font": {"size": 14}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                           "steps": [{"range": [0, umbral * 100], "color": "#EEF8F6"},
                                     {"range": [umbral * 100, 100], "color": "#FDF2F7"}],
                           "threshold": {"line": {"color": "#3B1D2B", "width": 3},
                                         "value": umbral * 100}}))
                fig.update_layout(height=200, margin=dict(t=40, b=0, l=20, r=20),
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")

            st.subheader("¿Por qué este resultado?")
            st.caption("Cada barra muestra cuánto pesó esa medida en la decisión. Se calcula "
                       "reemplazando la medida por su valor típico y viendo cuánto cambia la "
                       "predicción. Rosa = empuja hacia maligno · Verde = empuja hacia benigno.")
            exp = pd.DataFrame(pred["explicacion"]).iloc[::-1]
            exp["texto"] = exp.apply(
                lambda r: f"{r['peso_porcentaje']:.1f}% · valor {r['valor']:g} (típico {r['valor_tipico']:g})",
                axis=1)
            fig = go.Figure(go.Bar(
                x=exp["peso_porcentaje"], y=exp["nombre"], orientation="h",
                marker_color=[COLOR_MALIGNO if d == "maligno" else COLOR_BENIGNO
                              for d in exp["direccion"]],
                text=exp["texto"], textposition="outside", cliponaxis=False))
            fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=200),
                              xaxis_title="Peso en la decisión (%)", yaxis_title="",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Modelo: {pred['modelo']} ({pred['modelo_id']}). {pred['aviso']}")

elif pagina == "Historial":
    st.title("Historial de entrenamientos")
    historial, _ = api("GET", "/historial")
    if not historial:
        st.info("Todavía no hay entrenamientos.")
    else:
        filas = [{"id": h["id"], "algoritmo": h["nombre_algoritmo"],
                  "parámetros": str(h["parametros"]), **h["metricas"]} for h in historial]
        df = pd.DataFrame(filas[::-1])
        st.dataframe(df, width="stretch", hide_index=True)
        fig = px.bar(df.iloc[::-1], x="id", y=["sensibilidad", "f1", "roc_auc"], barmode="group",
                     labels={"value": "valor", "variable": "métrica"})
        fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis_range=[0.8, 1])
        st.plotly_chart(fig, width="stretch")
