"""
Copiloto de evaluación crediticia · Equipo 13 · Parte 2
Traduce la salida del modelo de scoring (XGBoost + SHAP) en explicaciones para
el analista, el comité y la PYME. Plantillas deterministas: misma entrada, mismo texto.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

AQUI = Path(__file__).parent

VARIABLES = {
    "puntualidad_pago_proveedores_pct": dict(
        nombre="Puntualidad de pago a proveedores", corto="tu puntualidad de pago", mejor="alto",
        fuente="Historial de pagos a proveedores", fmt=lambda v: f"{v:.0f} % de pagos a tiempo",
        consejo="pagar a tiempo a tus proveedores durante los próximos meses"),
    "ratio_endeudamiento": dict(
        nombre="Endeudamiento (deuda / ingresos)", corto="tu nivel de deuda", mejor="bajo",
        fuente="Estados de cuenta (agregador con consentimiento)", fmt=lambda v: f"{v:.2f}",
        consejo="reducir tu deuda frente a tus ingresos"),
    "antiguedad_negocio_anios": dict(
        nombre="Antigüedad del negocio", corto="la antigüedad de tu negocio", mejor="alto",
        fuente="CFDI (primer comprobante emitido)", fmt=lambda v: f"{v:.1f} años",
        consejo="acumular más historial de operación"),
    "variabilidad_flujo_cv": dict(
        nombre="Variabilidad del flujo mensual (CV)", corto="la estabilidad de tus ingresos", mejor="bajo",
        fuente="CFDI emitidos", fmt=lambda v: f"{v:.2f}",
        consejo="estabilizar tus ingresos mes a mes"),
    "ventas_anuales_mxn": dict(
        nombre="Ventas anuales", corto="tus ventas anuales", mejor="alto",
        fuente="CFDI emitidos", fmt=lambda v: f"${v / 1e6:,.1f} M MXN",
        consejo="mantener o aumentar tu facturación"),
    "concentracion_clientes_top3": dict(
        nombre="Concentración en 3 clientes", corto="la variedad de tus clientes", mejor="bajo",
        fuente="CFDI emitidos", fmt=lambda v: f"{v:.0%} de las ventas",
        consejo="diversificar tu cartera de clientes"),
}
FEATURES = list(VARIABLES)

GIROS = ["Taller de manufactura", "Comercializadora", "Distribuidora de alimentos", "Servicios logísticos",
         "Imprenta", "Carpintería", "Refaccionaria", "Agencia de marketing", "Laboratorio dental",
         "Panificadora", "Herrería", "Papelería mayorista", "Taller mecánico", "Constructora"]
APELLIDOS = ["Ramírez", "López", "Hernández", "García", "Martínez", "Torres", "Flores", "Morales",
             "Vázquez", "Castillo", "Ortega", "Ruiz", "Mendoza", "Navarro", "Cárdenas", "Salinas"]


def nombre_pyme(id_solicitud: int) -> str:
    """Nombre ficticio y determinista para mostrar en el prototipo (datos sintéticos)."""
    return f"{GIROS[(id_solicitud * 5 + 3) % len(GIROS)]} {APELLIDOS[(id_solicitud * 7 + 1) % len(APELLIDOS)]}"


class Copiloto:
    def __init__(self, carpeta: Path = AQUI / "modelo"):
        self.meta = json.loads((carpeta / "metadatos.json").read_text())
        self.modelo = XGBClassifier()
        self.modelo.load_model(str(carpeta / "modelo_xgb.json"))
        self.explainer = shap.TreeExplainer(self.modelo)
        self.mediana = pd.Series(self.meta["medianas"])
        self.u_rev, self.u_rech = self.meta["umbral_revisar"], self.meta["umbral_rechazo"]

    # ------------------------------------------------------------------ scoring
    def indice(self, X: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(X[FEATURES])[:, 1]

    def banda(self, p: float) -> str:
        return "Rechazar" if p >= self.u_rech else ("Revisar" if p >= self.u_rev else "Aprobar")

    def evaluar(self, fila: pd.Series) -> dict:
        X = fila[FEATURES].to_frame().T.astype(float)
        p = float(self.indice(X)[0])
        sv = self.explainer.shap_values(X)[0]
        contrib = pd.Series(sv, index=FEATURES)
        return {"p": p, "indice": round(p * 100), "banda": self.banda(p), "shap": contrib,
                "valores": fila[FEATURES], "faltantes": [f for f in FEATURES if pd.isna(fila[f])]}

    # ------------------------------------------------------------------ textos
    def _razon(self, f: str, valor, s: float) -> str:
        v = VARIABLES[f]
        if pd.isna(valor):
            return f"{v['nombre']}: sin dato; el modelo lo trata como faltante ({'sube' if s > 0 else 'baja'} el riesgo)."
        tipico = v["fmt"](self.mediana[f])
        efecto = "sube el riesgo" if s > 0 else "baja el riesgo"
        return f"{v['nombre']}: {v['fmt'](valor)} (típico: {tipico}) → {efecto}."

    def razones(self, ev: dict, n: int = 3) -> list[str]:
        top = ev["shap"].abs().sort_values(ascending=False).index[:n]
        return [self._razon(f, ev["valores"][f], ev["shap"][f]) for f in top]

    def que_necesita(self, ev: dict, tu: bool = False) -> str:
        """Contrafactual simple: menor cambio en puntualidad y/o endeudamiento que lleva a 'Aprobar'."""
        if ev["banda"] == "Aprobar":
            return "La solicitud ya está en la zona de aprobación; no se requiere ningún cambio."
        base = ev["valores"].astype(float).copy()
        opciones = []
        for dp in np.arange(0, 41, 5):
            for de in np.arange(0, 0.61, 0.05):
                x = base.copy()
                if not pd.isna(x["puntualidad_pago_proveedores_pct"]):
                    x["puntualidad_pago_proveedores_pct"] = min(100.0, x["puntualidad_pago_proveedores_pct"] + dp)
                x["ratio_endeudamiento"] = max(0.0, x["ratio_endeudamiento"] - de)
                if self.indice(x.to_frame().T)[0] < self.u_rev:
                    opciones.append((dp / 10 + de / 0.1, dp, de))
                    break
        if not opciones:
            return ("Con los datos actuales ningún cambio razonable en puntualidad o endeudamiento la lleva a "
                    "aprobación; conviene revisar en 6 meses con más historial.")
        _, dp, de = min(opciones)
        pos = "tu" if tu else "su"
        partes = []
        if dp:
            partes.append(f"subir {pos} puntualidad de pago a proveedores de {base['puntualidad_pago_proveedores_pct']:.0f} % a "
                          f"{min(100, base['puntualidad_pago_proveedores_pct'] + dp):.0f} %")
        if de:
            partes.append(f"bajar {pos} deuda frente a {'tus' if tu else 'sus'} ingresos de "
                          f"{base['ratio_endeudamiento']:.0%} a {max(0, base['ratio_endeudamiento'] - de):.0%}")
        if tu:
            return "Te recomendamos " + " y ".join(partes) + "; con eso tu solicitud tendría muy buenas posibilidades."
        return "Para pasar a la zona de aprobación necesitaría " + " y ".join(partes) + "."

    def faltantes(self, ev: dict) -> str:
        if not ev["faltantes"]:
            return "No faltan datos: las seis variables se obtuvieron de CFDI, estados de cuenta e historial de pagos."
        textos = []
        for f in ev["faltantes"]:
            if f == "concentracion_clientes_top3":
                textos.append("Concentración de clientes: no disponible (falta en ~65 % de las PYMES; peso bajo en el modelo).")
            elif f == "variabilidad_flujo_cv":
                textos.append("Variabilidad del flujo: menos de 6 meses de CFDI. Recomiendo revisión manual del flujo.")
            else:
                textos.append(f"{VARIABLES[f]['nombre']}: no disponible.")
        return " ".join(textos)

    def resumen_analista(self, ev: dict) -> str:
        r = "\n".join(f"- {x}" for x in self.razones(ev))
        return (f"**Recomendación: {ev['banda']}** · índice de riesgo {ev['indice']}/100 "
                f"(revisar desde {self.u_rev * 100:.0f}, rechazar desde {self.u_rech * 100:.0f}).\n\n"
                f"Lo que más pesó:\n{r}\n\n{self.faltantes(ev)}")

    def resumen_comite(self, ev: dict, nombre: str) -> str:
        r = "; ".join(x.rstrip(".") for x in self.razones(ev))
        return (f"**{nombre}** · recomendación del copiloto: **{ev['banda']}** (índice {ev['indice']}/100). "
                f"Factores principales: {r}. {self.que_necesita(ev)} "
                f"Nota: el modelo tiene AUC-ROC ≈ {self.meta['auc_prueba']:.2f} con datos sintéticos; "
                "es apoyo a la decisión, no la sustituye.")

    def mensaje_pyme(self, ev: dict, contacto: str = "") -> str:
        saludo = f"Hola {contacto}:" if contacto else "Hola:"
        estado = {"Aprobar": "tu solicitud fue preaprobada y pasa a firma",
                  "Revisar": "tu solicitud está en revisión con un analista",
                  "Rechazar": "por ahora no podemos aprobar tu solicitud"}[ev["banda"]]
        top = ev["shap"].sort_values(ascending=False)
        negativos = [VARIABLES[f] for f in top.index if top[f] > 0][:2]
        positivos = [VARIABLES[f]["corto"] for f in top.index[::-1] if top[f] < 0][:2]
        texto = f"{saludo} revisamos tu solicitud con tus facturas (CFDI) y tus pagos, y {estado}."
        if positivos:
            texto += f" A tu favor pesó {' y '.join(positivos)}."
        if negativos and ev["banda"] != "Aprobar":
            texto += " Lo que más te ayudaría: " + " y ".join(v["consejo"] for v in negativos) + "."
        return texto

    def responder(self, pregunta: str, ev: dict, nombre: str, contacto: str = "") -> str:
        q = pregunta.lower()
        if re.search(r"por ?qu[eé]|raz[oó]n|motivo|factor", q):
            return "Lo que más pesó en esta solicitud:\n" + "\n".join(f"- {x}" for x in self.razones(ev, 4))
        if re.search(r"mejor|calific|necesit|cambi|aprob", q):
            return self.que_necesita(ev)
        if re.search(r"comit[eé]|resumen|dictamen", q):
            return self.resumen_comite(ev, nombre)
        if re.search(r"pyme|cliente|mensaje|due[nñ]o|redact", q):
            return self.mensaje_pyme(ev, contacto)
        if re.search(r"falta|faltante|dato|document", q):
            return self.faltantes(ev)
        return ("Puedo responder: ¿por qué este resultado?, ¿qué necesita para calificar?, "
                "resumen para el comité, mensaje para la PYME y ¿qué datos faltan?")
