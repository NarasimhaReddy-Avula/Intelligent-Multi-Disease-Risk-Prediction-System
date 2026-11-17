# 🏥 MedRisk AI — Intelligent Multi-Disease Risk Prediction System

> **Bachelor Thesis Project (BTP) 2025–2026**
> IIIT Sri City · B.Tech CSE · Supervisor: Dr. R. Selvi
> Team: Avula Lakshmi Narasimha Reddy (S20230010030) · Rayapu Nishanth (S20230010202)

Predicts risk for **Diabetes, Heart Disease, Liver Disease and Kidney Disease** from a unified
24-feature clinical schema (~191K combined samples across four public datasets).

**Deployed model: per-disease XGBoost classifiers** (sentinel `-999` missing-value encoding),
served identically by the Streamlit app, the FastAPI service, and the PDF report generator
through the shared engine in `src/inference/engine.py`.

> ⚠️ For research/educational use only — not a medical device. See disclaimers in the app and reports.

---

## ✅ What's In The Box

| Component | Status | Location |
|---|---|---|
| Shared inference engine | ✅ | `src/inference/engine.py` — artifacts, prediction, TreeSHAP factors |
| Streamlit app | ✅ | `webapp/app.py` — inputs, risk cards, Plotly SHAP, history NLP, PDF download, LLM chat |
| FastAPI service | ✅ | `api/main.py` — `/predict` `/explain` `/report` `/extract-history` `/features` `/health` |
| PDF patient report | ✅ | `src/report/pdf_report.py` — 6-page report: all 4 diseases, drivers, cautions, doctors, diet, red flags |
| NLP lifestyle extractor | ✅ | `src/nlp/clinical_text_extractor.py` — enriches explanations/reports (not model inputs) |
| XGBoost artifacts | ✅ | `week7_artifacts/` — scaler + 4 models (mean test AUC ≈ 0.927) |
| Classical baselines | ✅ | `models/ml/` — LR/RF/XGB/LGBM per disease |
| FT-Transformer (experimental) | ⚠️ research | `models/dl/week7_ft_transformer/` — see note below |
| Tests | ✅ | `tests/test_system.py` — 14 tests |

### ⚠️ Experimental artifacts
`webapp/app2.py` and `models/dl/week7_ft_transformer/` are a **comparison build only**.
The FT-Transformer checkpoints came from inconsistent experiment runs; XGBoost was finalized
as the deployment model. Do not demo `app2.py` or cite FT numbers without re-running training.

---

## 🚀 Quick Start (macOS / Linux)

```bash
# Python 3.11+ required (works on 3.11–3.12)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt      # pinned, verified set

# Streamlit app
python -m streamlit run webapp/app.py    # → http://localhost:8501

# FastAPI service (separate terminal)
python -m uvicorn api.main:app --port 8000   # → http://localhost:8000/docs

# Tests
python -m pytest tests/test_system.py -q
```

Windows: `python -m venv .venv`, `.venv\Scripts\activate`, same commands after that.
`activate_env.bat` / `run_with_env.bat` are legacy helpers for the old Windows env.

### Optional: online LLM explanations
Set env vars (OpenAI-compatible; Groq shown as example):

```bash
export BTP_LLM_API_KEY=gsk_...
export BTP_LLM_BASE_URL=https://api.groq.com/openai/v1
export BTP_LLM_MODEL=openai/gpt-oss-20b
```

Without a key the app/API still work — the LLM section shows a rule-based fallback.
See `.env.example`.

---

## 🧪 Using The App

1. Enter patient age/gender in the sidebar.
2. Tick **"Enter …"** only for lab tests you actually have — everything else is
   auto-handled via sentinel encoding (shown transparently as "not provided").
3. *(Optional)* Paste a **patient history** — smoking/alcohol/activity/family-history
   are extracted and shown as chips, added to LLM context and the PDF. They do **not**
   change model inputs.
4. Press **Predict** → risk cards with marker-coverage chips, per-disease SHAP driver
   charts, cross-disease notes, the **cross-disease signal analysis** (attribution +
   ablation), global SHAP, and the **PDF report download**.

---

## 🔌 API Examples

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "features": {"age":55,"gender":1,"bmi":30,"hba1c":7.2,"glucose_fasting":140}}'

curl -X POST localhost:8000/explain -H 'Content-Type: application/json' -d '{
  "features": {"hba1c":7.2},"disease":"diabetes_risk","top_k":5}'

curl -X POST localhost:8000/report -H 'Content-Type: application/json' \
  -d '{"features":{"hba1c":7.2},"patient_name":"R. Sharma",
       "history_text":"smokes daily, father had diabetes"}' -o report.pdf
```

Full interactive docs at `/docs` (Swagger) when the server runs.

---

## 📊 Data & Features

Unified schema (`configs/unified_schema.yaml`): **24 clinical features** —
demographics (age, gender, BMI), vitals (systolic/diastolic BP), lipids
(cholesterol, HDL, LDL, triglycerides), glucose markers (fasting glucose, HbA1c),
liver panel (ALT, AST, ALP, albumin, total protein, bilirubin total/direct, A/G ratio),
kidney panel (creatinine, BUN, GFR, urine protein), hemoglobin.

Disease-cohort missingness (e.g. liver labs absent for heart-cohort rows) is encoded as
sentinel `-999` for XGBoost; a masked variant (value + availability flag, ~51 cols) exists
for the experimental FT-Transformer. **Patient-side missing inputs reuse the same encoding** —
unmeasured labs are left blank rather than imputed.

Sources: diabetes (~100K), heart (~70K), liver (~30.7K), kidney (~1.7K→~2K SMOTE) →
191,363 rows after dedup, 70/15/15 stratified split, seed 42.

**Risk bands (app/report):** 🟢 LOW <35% · 🟡 MEDIUM 35–65% · 🔴 HIGH >65%.

### Per-disease marker coverage & cross-disease scoring (important)
Each model was trained **only on its own cohort's rows** (e.g. kidney ≈ 2K samples
with all renal markers present). The system **always returns a score** — fused
cross-disease inference is the project's thesis — but qualifies confidence via
each disease's lab panel:

- **Diabetes:** fasting glucose or HbA1c
- **Heart:** BP or any lipid
- **Liver:** any of ALT/AST/ALP/bilirubin/albumin/protein/A-G ratio
- **Kidney:** creatinine/BUN/GFR/urine protein/hemoglobin

With no panel markers the score is shown **with a cross-disease confidence flag**:
app card caption + `n/m markers` coverage chip, PDF amber box, API
`sufficient_markers=false` + `confidence_note` naming the missing markers.
This surfaces genuine cross-disease signal (diabetes/CV markers do drive the
kidney score) while flagging the residual sentinel-extrapolation caveat.

### Cross-disease signal analysis
- `cross_disease_attribution()` — a disease's SHAP contributions summed by feature
  family (Demographics/Cardiovascular/Diabetes/Liver/Kidney): shows *which other
  diseases' data* drives each score.
- `panel_ablation()` — same patient scored three ways: all features vs. own
  markers only vs. markers removed → quantifies the panel's contribution vs
  cross-disease transfer. Rendered as grouped bar + table in the app.

### Explanation layer
- **TreeSHAP** per-disease drivers (Plotly diverging bars in app, matplotlib in PDF)
- **NLP lifestyle extractor:** patient-history text → smoking/alcohol/activity/family-history
  → shown as chips, injected into the LLM payload and PDF, and mapped to rule-based
  *clinical context notes* ("smoking is a risk factor not in model features — score may be
  conservative"). Never changes model inputs.
- **Online LLM** (OpenAI-compatible, Groq default): overall multi-disease assessment or
  per-disease explanation + follow-up chat. Rule-based fallback when no key is set.

---

## 🗂️ Structure (active files)

```
api/main.py                  FastAPI service
src/inference/engine.py      shared prediction + SHAP engine
src/report/pdf_report.py     PDF report generator
src/nlp/clinical_text_extractor.py   lifestyle/history extractor (regex; optional PubMedBERT)
src/llm/                     online-LLM client, prompts, fallback
src/explainability/shap_payload.py   structured SHAP factor builder
webapp/app.py                Streamlit app (deployed build)
week7_artifacts/             scaler + 4 XGBoost models + metrics  ← production artifacts
datasets/                    combined_unified_{masked,sentinel}[_scaled].csv
configs/unified_schema.yaml  24-feature schema + normal ranges
tests/test_system.py         engine/PDF/NLP/API tests
requirements-app.txt         pinned runtime deps (verified)
```

`demo.py`, `train.py`, `test_app.py`, `templates/`, `static/`, `src/models/predictor.py`,
`src/explainability/explainer.py`, `webapp/app2.py` are **legacy/experimental** — kept for
reference, not part of the shipped path. Older planning docs (`PROGRESS_TRACKER.md`,
`START_HERE.md`, `NEXT_STEPS.md`, `IMPLEMENTATION_SUMMARY.md`) reflect Semester-1 state.

---

## 📈 Results Snapshot (test set)

| Disease | XGBoost AUC |
|---|---|
| Diabetes | 0.938 |
| Heart | 0.801 |
| Liver | 0.999 |
| Kidney | 0.971 |
| **Mean** | **0.927** |

Known limitations: probability calibration is imperfect on some tasks (diabetes ECE ≈ 0.19);
models are trained on public research datasets and may not generalise to all populations.
Risk scores are screening indicators, not diagnoses.

---

*BTP 2025–26 · IIIT Sri City*
