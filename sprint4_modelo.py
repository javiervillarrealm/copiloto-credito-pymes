"""
Proyecto Integrador · Equipo 13 · Sprint 4
Prueba de concepto del modelo de scoring crediticio alternativo para PYMES.

Qué hace (de punta a punta, sin intervención manual):
  1. Genera un dataset SINTÉTICO de 5,000 solicitudes de PYMES (semilla 13).
  2. Divide en entrenamiento (4,000) y prueba (1,000), estratificado.
  3. Calibra XGBoost con búsqueda aleatoria (40 combinaciones, 5-fold estratificado, AUC-ROC).
  4. Compara contra una regresión logística (línea base) con la misma validación cruzada.
  5. Evalúa en prueba: AUC-ROC, KS, Brier y matriz de confusión.
  6. Calcula la importancia de variables con SHAP.
  7. Corre la prueba de valor simulada: modelo vs. regla manual del comité.

Uso:  python sprint4_modelo.py            (≈ 1–2 min)
Salidas: dataset_sintetico_pymes.csv, resultados_sprint4.json, fig_roc.png,
         fig_confusion.png, fig_shap.png

IMPORTANTE: los datos son sintéticos. Las relaciones entre variables e
incumplimiento son supuestos del equipo, no datos reales de PYMES.
"""
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 13
N = 5000
t0 = time.time()
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. Dataset sintético
antiguedad = np.clip(rng.lognormal(np.log(5), 0.7, N), 0.5, 30)                       # años
ventas = np.clip(rng.lognormal(np.log(6e6) + 0.15 * np.log(antiguedad / 5), 0.8, N), 3e5, 2.5e8)  # MXN/año
puntualidad = np.clip(rng.beta(7, 2, N) * 100, 5, 100)                                 # % pagos a tiempo a proveedores
variabilidad = np.clip(rng.gamma(4, 0.09, N) + 0.15 / np.sqrt(antiguedad), 0.05, 2.0) # CV del flujo mensual (CFDI)
endeudamiento = np.clip(rng.beta(2, 4, N) * 1.2, 0, 1.2)                              # deuda / ingresos
concentracion = np.clip(rng.beta(3, 4, N), 0.05, 1.0)                                 # % ventas top-3 clientes

logit = (
    -1.85
    - 0.055 * (puntualidad - 78)
    + 2.4 * (endeudamiento - 0.4)
    - 0.55 * np.log(antiguedad / 5)
    + 1.6 * (variabilidad - 0.5)
    - 0.12 * np.log(ventas / 6e6)
    + 0.35 * (concentracion - 0.43)
    + 1.1 * np.maximum(endeudamiento - 0.6, 0) * np.maximum(variabilidad - 0.5, 0) * 4   # interacción no lineal
    + rng.normal(0, 0.85, N)                                                           # lo que el modelo no ve
)
incumple = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

# Datos faltantes realistas: concentración solo existe para ~35 % de las PYMES;
# variabilidad falta en ~5 % (negocios con menos de 6 meses de CFDI).
concentracion = np.where(rng.random(N) < 0.35, concentracion, np.nan)
variabilidad = np.where((antiguedad < 1.5) & (rng.random(N) < 0.6), np.nan, variabilidad)

df = pd.DataFrame({
    "id_solicitud": np.arange(1, N + 1),
    "puntualidad_pago_proveedores_pct": puntualidad.round(1),
    "ratio_endeudamiento": endeudamiento.round(3),
    "antiguedad_negocio_anios": antiguedad.round(1),
    "variabilidad_flujo_cv": np.round(variabilidad, 3),
    "ventas_anuales_mxn": ventas.round(-3),
    "concentracion_clientes_top3": np.round(concentracion, 3),
    "incumplimiento_90d": incumple,
})
df.to_csv("dataset_sintetico_pymes.csv", index=False)

FEATURES = [c for c in df.columns if c not in ("id_solicitud", "incumplimiento_90d")]
NOMBRES = {
    "puntualidad_pago_proveedores_pct": "Puntualidad de pago a proveedores",
    "ratio_endeudamiento": "Ratio de endeudamiento",
    "antiguedad_negocio_anios": "Antigüedad del negocio",
    "variabilidad_flujo_cv": "Variabilidad del flujo (CFDI)",
    "ventas_anuales_mxn": "Ventas anuales (CFDI)",
    "concentracion_clientes_top3": "Concentración de clientes",
}
X, y = df[FEATURES], df["incumplimiento_90d"]

# ---------------------------------------------------------------- 2. Partición
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=1000, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

# ---------------------------------------------------------------- 3. Calibración XGBoost
espacio = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "n_estimators": [100, 200, 300, 400, 500],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}
busqueda = []
for params in ParameterSampler(espacio, n_iter=40, random_state=SEED):
    m = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="auc",
                      random_state=SEED, n_jobs=4, tree_method="hist")
    s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
    busqueda.append({**params, "auc_cv_media": s.mean(), "auc_cv_std": s.std()})
busqueda = pd.DataFrame(busqueda).sort_values("auc_cv_media", ascending=False)
mejor = busqueda.iloc[0]
best_params = {k: (int(mejor[k]) if k in ("max_depth", "n_estimators") else float(mejor[k])) for k in espacio}

# ---------------------------------------------------------------- 4. Línea base
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
X_tr_lr = X_tr.assign(ventas_anuales_mxn=np.log(X_tr.ventas_anuales_mxn))
X_te_lr = X_te.assign(ventas_anuales_mxn=np.log(X_te.ventas_anuales_mxn))
lr_cv = cross_val_score(lr, X_tr_lr, y_tr, cv=cv, scoring="roc_auc")
lr.fit(X_tr_lr, y_tr)
auc_lr_test = roc_auc_score(y_te, lr.predict_proba(X_te_lr)[:, 1])

# ---------------------------------------------------------------- 5. Modelo final y evaluación en prueba
modelo = XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="auc",
                       random_state=SEED, n_jobs=4, tree_method="hist")
modelo.fit(X_tr, y_tr)
p_te = modelo.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, p_te)
fpr, tpr, thr = roc_curve(y_te, p_te)
k = int(np.argmax(tpr - fpr))
ks, umbral = float((tpr - fpr)[k] * 100), float(thr[k])
pred = (p_te >= umbral).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
# Brier con la probabilidad recalibrada a la tasa real (scale_pos_weight infla las probabilidades)
xgb_cal = XGBClassifier(**best_params, eval_metric="auc", random_state=SEED, n_jobs=4, tree_method="hist").fit(X_tr, y_tr)
brier = brier_score_loss(y_te, xgb_cal.predict_proba(X_te)[:, 1])

# ---------------------------------------------------------------- 6. SHAP
sv = shap.TreeExplainer(modelo).shap_values(X_te)
imp = pd.Series(np.abs(sv).mean(0), index=FEATURES)
imp_pct = (imp / imp.sum() * 100).sort_values(ascending=False)

# ---------------------------------------------------------------- 7. Prueba de valor (simulada)
# Regla manual del comité: puntos por antigüedad, endeudamiento y puntualidad.
# Rechaza la misma proporción de solicitudes que el modelo, para comparar en igualdad.
puntos_manual = (
    (X_te.antiguedad_negocio_anios >= 3).astype(int)
    + (X_te.ratio_endeudamiento <= 0.5).astype(int)
    + (X_te.puntualidad_pago_proveedores_pct >= 80).astype(int)
    + rng.normal(0, 0.6, len(X_te))          # criterio variable entre analistas (PP-2)
)
tasa_rechazo = pred.mean()
rechazo_manual = (puntos_manual <= np.quantile(puntos_manual, tasa_rechazo)).astype(int)
aprob_ok_modelo = float(((pred == 0) & (y_te.values == 0)).sum() / (pred == 0).sum() * 100)
aprob_ok_manual = float(((rechazo_manual == 0) & (y_te.values == 0)).sum() / (rechazo_manual == 0).sum() * 100)
malos_detectados_modelo = float(tp / (tp + fn) * 100)
malos_detectados_manual = float(((rechazo_manual == 1) & (y_te.values == 1)).sum() / (y_te == 1).sum() * 100)

# ---------------------------------------------------------------- Gráficas
NAVY, BLUE, RED, GRID = "#0F2A52", "#2E6DB4", "#B23A3A", "#D7DCE5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

fig, ax = plt.subplots(figsize=(5.04, 4.2), dpi=200)
ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"XGBoost  AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], ls="--", color="#9A9A94", lw=1, label="Azar  AUC = 0.5")
ax.vlines(fpr[k], fpr[k], tpr[k], color=RED, lw=1.5, label=f"KS = {ks:.1f}")
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title(f"Curva ROC · conjunto de prueba (n = {len(y_te):,})", fontsize=11)
ax.legend(loc="lower right", frameon=False); ax.grid(color=GRID, lw=0.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("fig_roc.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.77, 4.2), dpi=200)
cm = np.array([[tn, fp], [fn, tp]])
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15,
                color="white" if cm[i, j] > cm.max() / 2 else NAVY)
ax.set_xticks([0, 1], ["Paga", "Incumple"]); ax.set_yticks([0, 1], ["Paga", "Incumple"])
ax.set_xlabel("Predicción del modelo"); ax.set_ylabel("Resultado real (90 días)")
ax.set_title(f"Matriz de confusión · umbral {umbral:.3f}", fontsize=11)
fig.tight_layout(); fig.savefig("fig_confusion.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.35, 4.2), dpi=200)
s = imp_pct.sort_values()
ax.barh([NOMBRES[i] for i in s.index], s.values, color=NAVY)
for i, v in enumerate(s.values): ax.text(v + 0.6, i, f"{v:.0f}%", va="center", fontsize=9)
ax.set_xlabel("Importancia SHAP (% del |SHAP| medio)")
ax.set_title("Importancia de variables (SHAP)", fontsize=11)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_xlim(0, s.max() * 1.18)
fig.tight_layout(); fig.savefig("fig_shap.png"); plt.close(fig)

# ---------------------------------------------------------------- Resultados
res = {
    "n_solicitudes": N, "tasa_incumplimiento_pct": round(float(y.mean() * 100), 1),
    "n_entrenamiento": len(y_tr), "n_prueba": len(y_te),
    "faltantes_pct": {c: round(float(df[c].isna().mean() * 100), 1) for c in FEATURES},
    "mejores_hiperparametros": best_params, "scale_pos_weight": round(spw, 2),
    "auc_cv_xgb": round(float(mejor.auc_cv_media), 3), "auc_cv_xgb_std": round(float(mejor.auc_cv_std), 3),
    "auc_cv_logistica": round(float(lr_cv.mean()), 3), "auc_cv_logistica_std": round(float(lr_cv.std()), 3),
    "auc_test_xgb": round(float(auc), 3), "auc_test_logistica": round(float(auc_lr_test), 3),
    "ks": round(ks, 1), "umbral": round(umbral, 3), "brier": round(float(brier), 3),
    "matriz_confusion": {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)},
    "importancia_shap_pct": {NOMBRES[k_]: round(float(v), 1) for k_, v in imp_pct.items()},
    "prueba_valor": {
        "tasa_rechazo_pct": round(float(tasa_rechazo * 100), 1),
        "aprobaciones_correctas_modelo_pct": round(aprob_ok_modelo, 1),
        "aprobaciones_correctas_manual_pct": round(aprob_ok_manual, 1),
        "incumplimientos_detectados_modelo_pct": round(malos_detectados_modelo, 1),
        "incumplimientos_detectados_manual_pct": round(malos_detectados_manual, 1),
    },
    "top5_busqueda": busqueda.head(5).round(4).to_dict("records"),
    "segundos": round(time.time() - t0, 1),
}
json.dump(res, open("resultados_sprint4.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k_: v for k_, v in res.items() if k_ != "top5_busqueda"}, indent=2, ensure_ascii=False))
