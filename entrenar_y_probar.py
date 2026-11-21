"""
Proyecto Integrador · Equipo 13 · Parte 2 (prototipado y prueba)

Parte del dataset sintético del Sprint 4 (dataset_sintetico_pymes.csv, semilla 13) y:
  1. Itera el modelo: XGBoost del Sprint 4 vs. XGBoost con restricciones monotónicas vs. regresión logística
     (misma partición, 5-fold estratificado, búsqueda aleatoria de 40 combinaciones).
  2. Guarda el modelo elegido para el prototipo (modelo/modelo_xgb.json + metadatos.json + bandeja.csv).
  3. Auditoría de sesgo por segmento (antigüedad y tamaño).
  4. Experimento factorial 2x2 con bloques: método (comité manual vs. copiloto) x segmento (joven vs. consolidada),
     30 réplicas bootstrap, ANOVA de dos vías.
  5. Latencia medida del prototipo (score + SHAP + textos del copiloto) por solicitud.

Uso: uv run --with-requirements requirements.txt python entrenar_y_probar.py      (≈ 3 min)
Datos sintéticos: las relaciones son supuestos del equipo, no datos reales de PYMES.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.anova import anova_lm
from xgboost import XGBClassifier

AQUI = Path(__file__).parent
FIG = AQUI.parent / "analisis"
MOD = AQUI / "modelo"
FIG.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)
SEED = 13
t0 = time.time()

df = pd.read_csv(AQUI / "dataset_sintetico_pymes.csv")
FEATURES = ["puntualidad_pago_proveedores_pct", "ratio_endeudamiento", "antiguedad_negocio_anios",
            "variabilidad_flujo_cv", "ventas_anuales_mxn", "concentracion_clientes_top3"]
X, y = df[FEATURES], df["incumplimiento_90d"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=1000, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

# ------------------------------------------------------------------ 1. Iteración del modelo
espacio = {"max_depth": [3, 4, 5, 6, 7, 8], "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
           "n_estimators": [100, 200, 300, 400, 500], "subsample": [0.7, 0.8, 0.9, 1.0],
           "colsample_bytree": [0.7, 0.8, 0.9, 1.0]}
S4 = dict(max_depth=3, learning_rate=0.01, n_estimators=500, subsample=0.9, colsample_bytree=0.7)  # Sprint 4
MONO = "(-1,1,-1,1,-1,1)"  # más puntualidad/antigüedad/ventas = menos riesgo; más deuda/variabilidad/concentración = más


def xgb(**kw):
    return XGBClassifier(**kw, scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=4, tree_method="hist")


s4_cv = cross_val_score(xgb(**S4), X_tr, y_tr, cv=cv, scoring="roc_auc")
busq = []
for params in ParameterSampler(espacio, n_iter=40, random_state=SEED):
    s = cross_val_score(xgb(**params, monotone_constraints=MONO), X_tr, y_tr, cv=cv, scoring="roc_auc")
    busq.append({**params, "auc_cv_media": s.mean(), "auc_cv_std": s.std()})
busq = pd.DataFrame(busq).sort_values("auc_cv_media", ascending=False)
mono_best = {k: (int(busq.iloc[0][k]) if k in ("max_depth", "n_estimators") else float(busq.iloc[0][k])) for k in espacio}
mono_cv = cross_val_score(xgb(**mono_best, monotone_constraints=MONO), X_tr, y_tr, cv=cv, scoring="roc_auc")

lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
log = lambda d: d.assign(ventas_anuales_mxn=np.log(d.ventas_anuales_mxn))
lr_cv = cross_val_score(lr, log(X_tr), y_tr, cv=cv, scoring="roc_auc")
lr.fit(log(X_tr), y_tr)

m_s4 = xgb(**S4).fit(X_tr, y_tr)
m_mono = xgb(**mono_best, monotone_constraints=MONO).fit(X_tr, y_tr)
p_s4, p_mono, p_lr = m_s4.predict_proba(X_te)[:, 1], m_mono.predict_proba(X_te)[:, 1], lr.predict_proba(log(X_te))[:, 1]

rng = np.random.default_rng(SEED)
boot = {"xgb_s4": [], "xgb_mono": [], "logistica": [], "dif_mono_vs_log": []}
yv = y_te.values
for _ in range(1000):
    i = rng.integers(0, len(yv), len(yv))
    if yv[i].min() == yv[i].max():
        continue
    a, b, c = roc_auc_score(yv[i], p_s4[i]), roc_auc_score(yv[i], p_mono[i]), roc_auc_score(yv[i], p_lr[i])
    boot["xgb_s4"].append(a); boot["xgb_mono"].append(b); boot["logistica"].append(c); boot["dif_mono_vs_log"].append(b - c)
ic = {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in boot.items()}

modelos = {
    "XGBoost Sprint 4": dict(cv=s4_cv, test=roc_auc_score(yv, p_s4)),
    "XGBoost monotónico (Parte 2)": dict(cv=mono_cv, test=roc_auc_score(yv, p_mono)),
    "Regresión logística (línea base)": dict(cv=lr_cv, test=roc_auc_score(yv, p_lr)),
}
elegido = "XGBoost monotónico (Parte 2)" if mono_cv.mean() >= s4_cv.mean() else "XGBoost Sprint 4"
modelo, p_te = (m_mono, p_mono) if elegido.startswith("XGBoost mono") else (m_s4, p_s4)

# ------------------------------------------------------------------ 2. Modelo para el prototipo
fpr, tpr, thr = roc_curve(yv, p_te)
k = int(np.argmax(tpr - fpr))
ks, u_rech = float((tpr - fpr)[k] * 100), float(thr[k])
tasa_rech = float((p_te >= u_rech).mean())
u_rev = float(np.quantile(p_te, 1 - tasa_rech - 0.15))  # ~15 % de las solicitudes van a revisión humana
pred = (p_te >= u_rech).astype(int)
vp, fp_, fn, vn = int(((pred == 1) & (yv == 1)).sum()), int(((pred == 1) & (yv == 0)).sum()), \
    int(((pred == 0) & (yv == 1)).sum()), int(((pred == 0) & (yv == 0)).sum())
modelo.save_model(str(MOD / "modelo_xgb.json"))
bandeja = df.loc[X_te.index].copy()
bandeja.to_csv(MOD / "bandeja.csv", index=False)
meta = {"modelo": elegido, "hiperparametros": mono_best if "mono" in elegido else S4,
        "monotonic": MONO if "mono" in elegido else None, "scale_pos_weight": round(spw, 2),
        "umbral_rechazo": round(u_rech, 4), "umbral_revisar": round(u_rev, 4),
        "auc_prueba": round(float(roc_auc_score(yv, p_te)), 3), "ks": round(ks, 1),
        "medianas": X_tr.median().round(3).to_dict()}
(MOD / "metadatos.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

# ------------------------------------------------------------------ 3. Auditoría por segmento
te = X_te.assign(y=yv, p=p_te, rech=pred, aprob=1 - pred)
te["antiguedad"] = pd.cut(te.antiguedad_negocio_anios, [0, 2, 5, 100], labels=["< 2 años", "2 a 5 años", "> 5 años"])
te["tamano"] = pd.qcut(te.ventas_anuales_mxn, 3, labels=["Ventas bajas", "Ventas medias", "Ventas altas"])
seg = []
for col in ("antiguedad", "tamano"):
    for g, d in te.groupby(col, observed=True):
        seg.append({"dimension": col, "segmento": str(g), "n": len(d), "incumplimiento_real_pct": round(d.y.mean() * 100, 1),
                    "auc": round(roc_auc_score(d.y, d.p), 3), "aprobacion_pct": round(d.aprob.mean() * 100, 1),
                    "incumplimiento_en_aprobados_pct": round(d[d.aprob == 1].y.mean() * 100, 1)})
seg = pd.DataFrame(seg)
razon_aprob = {c: round(float(s.aprobacion_pct.min() / s.aprobacion_pct.max()), 2) for c, s in seg.groupby("dimension")}

# ------------------------------------------------------------------ 4. Experimento factorial 2x2 (bloques por segmento)
# Respuesta: % de incumplimientos detectados y % de aprobaciones correctas, con la MISMA tasa de rechazo.
# Comité manual = regla de experiencia del Sprint 4 + variabilidad entre analistas (ruido nuevo en cada réplica).
filas = []
Xa, ya, pa = X_te.reset_index(drop=True), pd.Series(yv), pd.Series(p_te)
for r in range(30):
    rr = np.random.default_rng(1000 + r)
    i = rr.integers(0, len(ya), len(ya))
    Xi, yi, pi = Xa.iloc[i].reset_index(drop=True), ya.iloc[i].reset_index(drop=True), pa.iloc[i].reset_index(drop=True)
    puntos = ((Xi.antiguedad_negocio_anios >= 3).astype(int) + (Xi.ratio_endeudamiento <= 0.5).astype(int)
              + (Xi.puntualidad_pago_proveedores_pct >= 80).astype(int) + rr.normal(0, 0.6, len(Xi)))
    rech_cop = (pi >= u_rech).astype(int)
    rech_man = (puntos <= np.quantile(puntos, rech_cop.mean())).astype(int)
    joven = Xi.antiguedad_negocio_anios < 3
    for metodo, rech in (("Comité manual", rech_man), ("Copiloto", rech_cop)):
        for seg_n, m in (("Joven (< 3 años)", joven), ("Consolidada (≥ 3 años)", ~joven)):
            yy, rc = yi[m], rech[m]
            filas.append({"replica": r, "metodo": metodo, "segmento": seg_n,
                          "detectados_pct": float(((rc == 1) & (yy == 1)).sum() / max(1, (yy == 1).sum()) * 100),
                          "aprob_correctas_pct": float(((rc == 0) & (yy == 0)).sum() / max(1, (rc == 0).sum()) * 100),
                          "aprobacion_pct": float((rc == 0).mean() * 100)})
fx = pd.DataFrame(filas)
anova = {}
for resp in ("detectados_pct", "aprob_correctas_pct"):
    a = anova_lm(smf.ols(f"{resp} ~ C(metodo) * C(segmento)", data=fx).fit(), typ=2)
    anova[resp] = {idx: {"F": round(float(row.F), 1), "p": float(f"{row['PR(>F)']:.2g}")} for idx, row in a.dropna().iterrows()}
celdas = fx.groupby(["metodo", "segmento"])[["detectados_pct", "aprob_correctas_pct", "aprobacion_pct"]].mean().round(1)

# ------------------------------------------------------------------ 5. Latencia medida del prototipo
from copiloto import Copiloto, nombre_pyme  # noqa: E402
cop = Copiloto(MOD)
lat = []
for _, fila in bandeja.head(200).iterrows():
    t = time.perf_counter()
    ev = cop.evaluar(fila)
    cop.resumen_analista(ev); cop.mensaje_pyme(ev); cop.que_necesita(ev)
    lat.append((time.perf_counter() - t) * 1000)

# ------------------------------------------------------------------ Gráficas
NAVY, BLUE, RED, GRAY, GRID = "#0F2A52", "#2E6DB4", "#B23A3A", "#9AA3AE", "#D7DCE5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
nombres = list(modelos)
med = [modelos[n]["cv"].mean() for n in nombres]
sd = [modelos[n]["cv"].std() for n in nombres]
cols = [GRAY, NAVY, BLUE]
ax.barh(nombres, med, xerr=sd, color=cols, capsize=4)
for i, (m_, s_) in enumerate(zip(med, sd)):
    ax.text(m_ + s_ + 0.003, i, f"{m_:.3f}", va="center", fontsize=9)
ax.axvline(0.75, color=RED, ls="--", lw=1); ax.text(0.751, 2.45, "meta 0.75", color=RED, fontsize=8)
ax.set_xlim(0.68, 0.79); ax.set_xlabel("AUC-ROC · validación cruzada 5-fold (media ± desv.)")
ax.invert_yaxis()
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig_iteracion_modelo.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200)
orden = [("Comité manual", "Joven (< 3 años)"), ("Copiloto", "Joven (< 3 años)"),
         ("Comité manual", "Consolidada (≥ 3 años)"), ("Copiloto", "Consolidada (≥ 3 años)")]
for ax, (resp, tit) in zip(axes, (("detectados_pct", "Incumplimientos detectados (%)"),
                                  ("aprob_correctas_pct", "Aprobaciones correctas (%)"))):
    data = [fx[(fx.metodo == m_) & (fx.segmento == s_)][resp] for m_, s_ in orden]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="white", lw=1.5))
    for patch, (m_, _) in zip(bp["boxes"], orden):
        patch.set_facecolor(GRAY if m_ == "Comité manual" else NAVY)
    ax.set_xticks([1, 2, 3, 4], ["Manual\njoven", "Copiloto\njoven", "Manual\nconsolidada", "Copiloto\nconsolidada"], fontsize=8)
    ax.set_title(tit, fontsize=10); ax.grid(axis="y", color=GRID, lw=0.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.suptitle("Experimento factorial 2×2 · 30 réplicas bootstrap del conjunto de prueba", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig_experimento_factorial.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)
for ax, dim, tit in zip(axes, ("antiguedad", "tamano"), ("Por antigüedad", "Por tamaño (ventas)")):
    s = seg[seg.dimension == dim]
    x = np.arange(len(s))
    ax.bar(x - 0.2, s.aprobacion_pct, 0.4, color=NAVY, label="Aprobación recomendada %")
    ax.bar(x + 0.2, s.incumplimiento_real_pct, 0.4, color=RED, label="Incumplimiento real %")
    for xi, (a_, b_) in enumerate(zip(s.aprobacion_pct, s.incumplimiento_real_pct)):
        ax.text(xi - 0.2, a_ + 1, f"{a_:.0f}", ha="center", fontsize=8); ax.text(xi + 0.2, b_ + 1, f"{b_:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{g}\nAUC {a:.2f}" for g, a in zip(s.segmento, s.auc)], fontsize=8)
    ax.set_title(tit, fontsize=10); ax.set_ylim(0, 105); ax.grid(axis="y", color=GRID, lw=0.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_auditoria_segmentos.png"); plt.close(fig)

# ------------------------------------------------------------------ Resultados
res = {
    "iteracion_modelo": {n: {"auc_cv_media": round(float(v["cv"].mean()), 3), "auc_cv_std": round(float(v["cv"].std()), 3),
                             "auc_prueba": round(float(v["test"]), 3)} for n, v in modelos.items()},
    "ic95_auc_prueba_bootstrap": ic, "elegido": elegido, "hiperparametros_monotonico": mono_best,
    "top5_busqueda_monotonico": busq.head(5).round(4).to_dict("records"),
    "prototipo": {**{k_: meta[k_] for k_ in ("umbral_rechazo", "umbral_revisar", "auc_prueba", "ks")},
                  "tasa_rechazo_pct": round(tasa_rech * 100, 1),
                  "tasa_revision_pct": round(float(((p_te >= u_rev) & (p_te < u_rech)).mean() * 100), 1),
                  "matriz_confusion": {"VP": vp, "FP": fp_, "FN": fn, "VN": vn}},
    "auditoria_segmentos": seg.to_dict("records"), "razon_aprobacion_min_max": razon_aprob,
    "experimento_factorial": {"celdas": {f"{m_} | {s_}": v for (m_, s_), v in celdas.to_dict("index").items()}, "anova": anova},
    "latencia_ms": {"mediana": round(float(np.median(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1), "n": len(lat)},
    "segundos": round(time.time() - t0, 1),
}
(FIG / "resultados_parte2.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
