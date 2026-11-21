"""
Prototipo MVP · Copiloto de evaluación crediticia para PYMES sin historial en buró
Equipo 13 · Proyecto Integrador Parte 2 · datos sintéticos

Correr:  uv run --with-requirements requirements.txt streamlit run app.py
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
from copiloto import FEATURES, VARIABLES, Copiloto, nombre_pyme  # noqa: E402

BITACORA = AQUI / "bitacora_decisiones.csv"
COLOR = {"Aprobar": "#1E7B4F", "Revisar": "#B7791F", "Rechazar": "#B23A3A"}
CONTACTOS = {}  # id_solicitud -> nombre del dueño, para la vista de la PYME (se llena en metadatos)

st.set_page_config(page_title="Copiloto de evaluación crediticia", page_icon="📊", layout="wide")


@st.cache_resource
def cargar():
    cop = Copiloto(AQUI / "modelo")
    b = pd.read_csv(AQUI / "modelo" / "bandeja.csv")
    p = cop.indice(b)
    sv = pd.DataFrame(cop.explainer.shap_values(b[FEATURES]), columns=FEATURES)
    b["indice"] = (p * 100).round().astype(int)
    b["recomendacion"] = [cop.banda(x) for x in p]
    b["pyme"] = b.id_solicitud.map(nombre_pyme)
    for i, nom in cop.meta.get("nombres_demo", {}).items():
        b.loc[b.id_solicitud == int(i), "pyme"] = nom
    b["folio"] = b.id_solicitud.map(lambda i: f"PYME-{i:04d}")
    top = sv.abs().idxmax(axis=1)
    b["factor_principal"] = [f"{VARIABLES[f]['nombre']} ({'↑' if sv.loc[i, f] > 0 else '↓'} riesgo)" for i, f in top.items()]
    b["antiguedad_grupo"] = pd.cut(b.antiguedad_negocio_anios, [0, 2, 5, 100], labels=["< 2 años", "2 a 5 años", "> 5 años"])
    return cop, b


cop, bandeja = cargar()
CONTACTOS.update({int(k): v for k, v in cop.meta.get("contactos_demo", {}).items()})
qp = st.query_params
vistas = ["Analista de crédito", "Comité de crédito", "Vista de la PYME", "Bitácora de decisiones"]
vista_qp = {"analista": 0, "comite": 1, "pyme": 2, "bitacora": 3}.get(qp.get("vista", ""), 0)

with st.sidebar:
    st.markdown("### Copiloto de evaluación crediticia")
    st.caption("Prototipo MVP · Equipo 13 · datos sintéticos")
    vista = st.radio("Vista", vistas, index=vista_qp)
    st.divider()
    st.caption(f"Modelo: {cop.meta['modelo']}  \nAUC-ROC prueba: {cop.meta['auc_prueba']:.3f} · KS: {cop.meta['ks']}")
    st.caption(f"Revisar desde índice {cop.u_rev * 100:.0f} · rechazar desde {cop.u_rech * 100:.0f}")

orden = bandeja.sort_values("indice", ascending=False)
etiquetas = {r.id_solicitud: f"{r.folio} · {r.pyme} · {r.recomendacion} ({r.indice})" for r in orden.itertuples()}
ids = list(etiquetas)
id_qp = int(qp.get("id", ids[0])) if str(qp.get("id", "")).isdigit() else ids[0]


def badge(banda: str) -> str:
    return (f"<span style='background:{COLOR[banda]};color:white;padding:4px 12px;border-radius:12px;"
            f"font-weight:600'>{banda}</span>")


def grafica_shap(ev: dict):
    d = pd.DataFrame({"Variable": [VARIABLES[f]["nombre"] for f in FEATURES], "Contribución": ev["shap"].values})
    d["Efecto"] = d["Contribución"].map(lambda v: "Sube el riesgo" if v > 0 else "Baja el riesgo")
    d = d.reindex(d["Contribución"].abs().sort_values(ascending=False).index)
    return (alt.Chart(d).mark_bar(cornerRadius=3).encode(
        x=alt.X("Contribución:Q", title="Contribución al índice de riesgo (SHAP)"),
        y=alt.Y("Variable:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False, labelFontSize=12)),
        color=alt.Color("Efecto:N", scale=alt.Scale(domain=["Sube el riesgo", "Baja el riesgo"], range=["#B23A3A", "#1E7B4F"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Variable", alt.Tooltip("Contribución:Q", format=".3f")]).properties(height=270))


# ====================================================================== Analista
if vista == "Analista de crédito":
    st.markdown("## Bandeja del analista · Mariana Torres")
    sel = st.selectbox("Solicitud", ids, index=ids.index(id_qp) if id_qp in ids else 0, format_func=etiquetas.get)
    fila = bandeja.set_index("id_solicitud").loc[sel]
    ev = cop.evaluar(fila)
    nombre = f"{fila.folio} · {fila.pyme}"

    c1, c2, c3 = st.columns(3)
    c1.metric("Índice de riesgo", f"{ev['indice']} / 100")
    c2.markdown("Recomendación del copiloto<br>" + badge(ev["banda"]), unsafe_allow_html=True)
    c3.metric("Datos disponibles", f"{6 - len(ev['faltantes'])} de 6")

    izq, der = st.columns([1.2, 1])
    with izq:
        st.markdown("#### Datos extraídos automáticamente")
        tabla = pd.DataFrame([{"Variable": VARIABLES[f]["nombre"],
                               "Valor": "sin dato" if pd.isna(fila[f]) else VARIABLES[f]["fmt"](fila[f]),
                               "Típico": VARIABLES[f]["fmt"](cop.mediana[f])} for f in FEATURES])
        st.dataframe(tabla, hide_index=True, use_container_width=True)
        st.caption("Fuentes: CFDI emitidos y recibidos, estados de cuenta (agregador con consentimiento de la PYME) "
                   "e historial de pagos a proveedores.")
    with der:
        st.markdown("#### ¿Qué pesó en el resultado?")
        st.altair_chart(grafica_shap(ev), use_container_width=True)

    st.markdown("#### Copiloto")
    hist = st.session_state.setdefault(f"chat_{sel}", [("assistant", cop.resumen_analista(ev))])
    preguntas = ["¿Por qué este resultado?", "¿Qué necesita para calificar?", "Resumen para el comité",
                 "Mensaje para la PYME", "¿Qué datos faltan?"]
    cols = st.columns(len(preguntas))
    for col, q in zip(cols, preguntas):
        if col.button(q, key=f"{sel}_{q}", use_container_width=True):
            hist += [("user", q), ("assistant", cop.responder(q, ev, nombre, CONTACTOS.get(int(sel), "").split(" ")[0]))]
    for q in qp.get_all("pregunta") if len(hist) == 1 else []:
        hist += [("user", q), ("assistant", cop.responder(q, ev, nombre, CONTACTOS.get(int(sel), "").split(" ")[0]))]
    for rol, txt in hist[-6:]:
        st.chat_message(rol).markdown(txt)
    if q := st.chat_input("Pregúntale al copiloto sobre esta solicitud"):
        hist += [("user", q), ("assistant", cop.responder(q, ev, nombre, CONTACTOS.get(int(sel), "").split(" ")[0]))]
        st.rerun()

    st.markdown("#### Dictamen del analista")
    d1, d2, d3 = st.columns([1, 2, 1])
    decision = d1.radio("Decisión", ["Aprobar", "Revisar", "Rechazar"], index=["Aprobar", "Revisar", "Rechazar"].index(ev["banda"]))
    motivo = d2.text_input("Motivo (obligatorio si difiere de la recomendación)")
    if d3.button("Registrar dictamen", type="primary", use_container_width=True):
        if decision != ev["banda"] and not motivo.strip():
            st.error("Escribe el motivo: la decisión difiere de la recomendación del copiloto.")
        else:
            nuevo = not BITACORA.exists()
            with BITACORA.open("a", newline="") as fh:
                w = csv.writer(fh)
                if nuevo:
                    w.writerow(["fecha", "folio", "pyme", "indice", "recomendacion", "decision", "coincide", "motivo"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), fila.folio, fila.pyme, ev["indice"],
                            ev["banda"], decision, decision == ev["banda"], motivo.strip()])
            st.success("Dictamen registrado en la bitácora.")

# ====================================================================== Comité
elif vista == "Comité de crédito":
    st.markdown("## Tablero del comité de crédito")
    n = len(bandeja)
    k = bandeja.recomendacion.value_counts()
    c = st.columns(4)
    c[0].metric("Solicitudes en bandeja", f"{n:,}")
    for col, b_ in zip(c[1:], ["Aprobar", "Revisar", "Rechazar"]):
        col.metric(f"{b_} · {k.get(b_, 0)} solicitudes", f"{k.get(b_, 0) / n:.0%}")

    izq, der = st.columns([1.6, 1])
    with izq:
        st.markdown("#### Solicitudes priorizadas por riesgo")
        filtro = st.multiselect("Recomendación", ["Rechazar", "Revisar", "Aprobar"], default=["Rechazar", "Revisar"])
        t = orden[orden.recomendacion.isin(filtro)][["folio", "pyme", "indice", "recomendacion", "factor_principal"]]
        st.dataframe(t.rename(columns={"folio": "Folio", "pyme": "PYME", "indice": "Índice", "recomendacion": "Recomendación",
                                       "factor_principal": "Factor principal"}),
                     hide_index=True, use_container_width=True, height=360)
    with der:
        st.markdown("#### Auditoría de sesgo")
        g = bandeja.groupby("antiguedad_grupo", observed=True).recomendacion.apply(lambda s: (s != "Rechazar").mean() * 100)
        aud = g.reset_index().rename(columns={"antiguedad_grupo": "Antigüedad", "recomendacion": "Aprobación %"})
        st.altair_chart(alt.Chart(aud).mark_bar(color="#0F2A52", cornerRadius=3).encode(
            x=alt.X("Antigüedad:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Aprobación %:Q", title="No rechazadas (%)"),
            tooltip=["Antigüedad", alt.Tooltip("Aprobación %:Q", format=".1f")]).properties(height=220), use_container_width=True)
        razon = g.min() / g.max()
        if razon < 0.8:
            st.warning(f"Razón de aprobación mín./máx. = {razon:.2f} (< 0.80, regla de 4/5). "
                       "Los negocios jóvenes se aprueban menos: revisar si la diferencia se explica por el riesgo real.")
        else:
            st.success(f"Razón de aprobación mín./máx. = {razon:.2f} (≥ 0.80).")
    st.markdown("#### Resumen para el comité")
    sel = st.selectbox("Solicitud", ids, index=ids.index(id_qp) if id_qp in ids else 0, format_func=etiquetas.get)
    fila = bandeja.set_index("id_solicitud").loc[sel]
    st.info(cop.resumen_comite(cop.evaluar(fila), f"{fila.folio} · {fila.pyme}"))

# ====================================================================== PYME
elif vista == "Vista de la PYME":
    sel = id_qp if id_qp in ids else ids[0]
    fila = bandeja.set_index("id_solicitud").loc[sel]
    ev = cop.evaluar(fila)
    contacto = CONTACTOS.get(int(sel), "").split(" ")[0]
    st.markdown(f"## Tu solicitud de crédito · {fila.pyme}")
    st.caption(f"Folio {fila.folio}" + (f" · {contacto}" if contacto else ""))
    pasos = ["Solicitud recibida", "Datos verificados (CFDI y estados de cuenta)", "Evaluación", "Dictamen"]
    cols = st.columns(4)
    for col, p_ in zip(cols[:3], pasos[:3]):
        col.success(f"✓ {p_}")
    if ev["banda"] == "Revisar":
        cols[3].warning("… Dictamen: en revisión")
    else:
        cols[3].success("✓ Dictamen")
    st.markdown("#### Resultado")
    st.markdown(badge({"Aprobar": "Aprobar", "Revisar": "Revisar", "Rechazar": "Rechazar"}[ev["banda"]])
                .replace(ev["banda"], {"Aprobar": "Preaprobada", "Revisar": "En revisión", "Rechazar": "No aprobada por ahora"}[ev["banda"]]),
                unsafe_allow_html=True)
    st.write("")
    st.info(cop.mensaje_pyme(ev, contacto.split()[0] if contacto else ""))
    if ev["banda"] != "Aprobar":
        st.markdown("#### ¿Qué puedes hacer?")
        st.write(cop.que_necesita(ev, tu=True))
    with st.expander("¿Qué datos usamos y cómo los cuidamos?"):
        st.write("Usamos tus facturas electrónicas (CFDI), tus estados de cuenta y tu historial de pagos a proveedores, "
                 "obtenidos con tu consentimiento a través de un agregador autorizado. Solo se usan para evaluar esta "
                 "solicitud y puedes revocar el consentimiento en cualquier momento.")

# ====================================================================== Bitácora
else:
    st.markdown("## Bitácora de decisiones")
    if BITACORA.exists():
        b = pd.read_csv(BITACORA)
        c1, c2 = st.columns(2)
        c1.metric("Dictámenes registrados", len(b))
        c2.metric("Coincidencia analista vs. copiloto", f"{b.coincide.astype(str).eq('True').mean():.0%}")
        st.dataframe(b.iloc[::-1], hide_index=True, use_container_width=True)
    else:
        st.info("Aún no hay dictámenes registrados. Registra uno desde la vista del analista.")
