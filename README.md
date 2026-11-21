# Copiloto de evaluación crediticia · prototipo (Equipo 13, Parte 2)

Prototipo funcional con datos sintéticos: modelo XGBoost monotónico + SHAP, un copiloto que explica cada
resultado con plantillas deterministas, y una app Streamlit con cuatro vistas (analista, comité, PYME, bitácora).

## Correr la app

```bash
cd prototipo
uv run --with-requirements requirements.txt streamlit run app.py
# o, con pip:  pip install -r requirements.txt && streamlit run app.py
```

Se abre en http://localhost:8501. Enlaces directos útiles para la demo (caso de Héctor, PYME-1809):

| Vista | URL |
|---|---|
| Analista | `http://localhost:8501/?vista=analista&id=1809` |
| Analista con preguntas al copiloto | `http://localhost:8501/?vista=analista&id=1809&pregunta=¿Qué necesita para calificar?&pregunta=Mensaje para la PYME` |
| Comité | `http://localhost:8501/?vista=comite` |
| PYME | `http://localhost:8501/?vista=pyme&id=1809` |
| Bitácora | `http://localhost:8501/?vista=bitacora` |

## Archivos

| Archivo | Qué hace |
|---|---|
| `sprint4_modelo.py` | Script de Sofía del Sprint 4 (genera `dataset_sintetico_pymes.csv` y el modelo base). |
| `entrenar_y_probar.py` | Iteración del modelo (Sprint 4 vs. monotónico vs. logística), umbrales, auditoría de sesgo, experimento factorial 2×2 con ANOVA y latencia. Escribe `modelo/` y `../analisis/`. |
| `figuras_modelo.py` | Curva ROC, pesos SHAP, matriz de confusión y casos de demostración. |
| `copiloto.py` | Índice, banda (Aprobar < 45 ≤ Revisar < 53 ≤ Rechazar), razones SHAP, contrafactual y textos. |
| `app.py` | Interfaz Streamlit; los dictámenes se guardan en `bitacora_decisiones.csv`. |

Orden para regenerar todo: `sprint4_modelo.py` → `entrenar_y_probar.py` → `figuras_modelo.py` → `app.py`.

Las cifras son de datos sintéticos (5,000 PYMES) y de una regla manual que simula al comité; no son de usuarios reales.
