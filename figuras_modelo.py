"""Figuras de evaluación del modelo elegido (Parte 2) y casos de demostración del prototipo."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
from copiloto import FEATURES, VARIABLES, Copiloto, nombre_pyme  # noqa: E402

FIG = AQUI.parent / "analisis"
cop = Copiloto(AQUI / "modelo")
b = pd.read_csv(AQUI / "modelo" / "bandeja.csv")
y = b.incumplimiento_90d.values
p = cop.indice(b)
sv = cop.explainer.shap_values(b[FEATURES])
imp = pd.Series(np.abs(sv).mean(0), index=FEATURES)
imp_pct = (imp / imp.sum() * 100).sort_values(ascending=False)
fpr, tpr, thr = roc_curve(y, p)
k = int(np.argmax(tpr - fpr))
pred = (p >= cop.u_rech).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

NAVY, RED, GRID = "#0F2A52", "#B23A3A", "#D7DCE5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=200)
ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"XGBoost monotónico  AUC = {roc_auc_score(y, p):.3f}")
ax.plot([0, 1], [0, 1], ls="--", color="#9A9A94", lw=1, label="Azar  AUC = 0.5")
ax.vlines(fpr[k], fpr[k], tpr[k], color=RED, lw=1.5, label=f"KS = {(tpr - fpr)[k] * 100:.1f}")
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curva ROC · prueba (n = 1,000)", fontsize=10)
ax.legend(loc="lower right", frameon=False, fontsize=8); ax.grid(color=GRID, lw=0.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig_roc_parte2.png"); plt.close(fig)

# Casos de demostración: Héctor (taller, en revisión, con cambio alcanzable) y un caso aprobado
ev_cols = []
for r in b.itertuples():
    ev_cols.append(cop.banda(p[r.Index]))
b["banda"] = ev_cols
b["pyme"] = b.id_solicitud.map(nombre_pyme)
cand = b[(b.banda == "Revisar") & b.concentracion_clientes_top3.notna() & b.variabilidad_flujo_cv.notna()
         & (b.antiguedad_negocio_anios.between(3, 12)) & (b.puntualidad_pago_proveedores_pct < 75)]
hector = None
for r in cand.itertuples():
    ev = cop.evaluar(b.loc[r.Index])
    if "necesitaría" in cop.que_necesita(ev):
        hector = int(r.id_solicitud); break
aprobado = int(b[(b.banda == "Aprobar")].sort_values("indice" if "indice" in b else "id_solicitud").id_solicitud.iloc[0])
meta_path = AQUI / "modelo" / "metadatos.json"
meta = json.loads(meta_path.read_text())
meta["contactos_demo"] = {str(hector): "Héctor Ramírez Cruz"}
meta["nombres_demo"] = {str(hector): "Taller de manufactura Ramírez"}
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

out = {"shap_pct": imp_pct.round(1).rename(lambda f: VARIABLES[f]["nombre"]).to_dict(),
       "matriz": {"VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn)},
       "precision_rechazos_pct": round(tp / (tp + fp) * 100, 1), "recall_pct": round(tp / (tp + fn) * 100, 1),
       "demo": {"hector": hector, "aprobado": aprobado,
                "hector_eval": cop.resumen_analista(cop.evaluar(b.set_index("id_solicitud").loc[hector])),
                "hector_pyme": cop.mensaje_pyme(cop.evaluar(b.set_index("id_solicitud").loc[hector]), "Héctor"),
                "hector_necesita": cop.que_necesita(cop.evaluar(b.set_index("id_solicitud").loc[hector]))},
       "bandas_pct": (b.banda.value_counts(normalize=True) * 100).round(1).to_dict()}
(FIG / "evaluacion_parte2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
