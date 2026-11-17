# Proposal → Delivery Coverage Map

*For the final report. Every initial proposal commitment, its delivery status, evidence,
and the justification for any deviation. Verified against the live system (2026-04-19).*

## 1. Disease coverage

| Proposed | Delivered | Status |
|---|---|---|
| Heart Disease | ✅ Heart Disease (`heart_disease_risk`) | Delivered — test AUC 0.801 |
| Diabetes | ✅ Diabetes (`diabetes_risk`) | Delivered — test AUC 0.938 |
| Breast Cancer | ❌ → replaced by Kidney Disease | **Deviation — justified below** |
| Liver Disease | ✅ Liver Disease (`liver_disease_risk`) | Delivered — test AUC 0.999 |
| (Sem-2 optional) extended coverage | ✅ Kidney Disease (`kidney_disease_risk`) | Delivered — test AUC 0.971 |

**Justification for the breast-cancer → kidney swap:** the Wisconsin Diagnostic dataset
is tumor-morphology (imaging-derived) features that share no schema overlap with
lab-test features — it cannot participate in *heterogeneous clinical data fusion*, which
is the project's core contribution. Kidney disease (the proposal's own "extended
coverage" option) shares glucose/BP/demographic features with the other cohorts and
enables the clinically meaningful cardiorenal / diabetic-nephropathy cross-disease
narrative. The swap strengthened, not weakened, the fusion thesis.

## 2. Data sources

| Proposed | Delivered | Note |
|---|---|---|
| Framingham (heart), Pima (diabetes), ILPD (liver), Wisconsin | Larger public cohorts: heart ~70K, diabetes ~100K, liver ~30.7K, kidney ~1.7K→~2K (SMOTE) | **Exceeds proposal** — 191,363 unified rows vs. ~few-K per proposal dataset |
| MIMIC-III clinical notes | Not used — credentialing timeline risk (proposal's own risk table anticipated this) | Patient-history free text handled by the NLP extractor instead |
| Unified schema, 36 features | 24-feature unified schema | Leaner, fully populated schema; sentinel `-999` + masked encodings implemented |

## 3. Technology stack

| Proposed | Delivered | Evidence |
|---|---|---|
| XGBoost/LightGBM baselines | LogisticReg + RandomForest + LightGBM + XGBoost per disease, both encodings (`models/ml/`, `week7_artifacts/xgb_baselines.csv`) | ✅ Delivered; XGBoost deployed |
| FT-Transformer | Trained, evaluated — `models/dl/`, `best_model.pt`, `train_history.csv` | ✅ Delivered as research artifact — test mean AUC 0.931 vs XGB 0.927; kept experimental (prior-run inconsistencies), XGBoost deployed for reliability — exactly the proposal's risk-mitigation path |
| NLP (BioMistral/PubMedBERT) | Rule-based clinical-text extractor (`clinical_text_extractor.py`) + optional PubMedBERT hook | ⚠️ Partial — extractor covers smoking/alcohol/activity/family-hx/symptoms/comorbidities; 10/10 phrase tests. No LLM-scale NLP — justified in §6 |
| LLM explanations (GPT-4/Llama) | OpenAI-compatible client, Groq `openai/gpt-oss-20b` default | ✅ Delivered — overall + per-disease explanations + follow-up chat with history; verified `llm_json` mode live; rule-based fallback without key — matches the free-tier mitigation plan |
| SHAP (XAI) | TreeSHAP per-patient + global importance | ✅ Delivered — diverging-bar charts in app, rendered charts in PDF; verified per-instance (not static) |
| Heterogeneous data fusion | 4-cohort fusion, unified 24-feature schema, sentinel/masked missingness | ✅ Delivered — **core contribution** + cross-disease attribution/ablation added beyond proposal |
| FastAPI + Streamlit/Gradio | FastAPI (6 endpoints, OpenAPI docs) + Streamlit app | ✅ Delivered; 7 ms predict latency vs. <500 ms target |
| PDF patient report | 6-page fpdf2 report (all diseases, drivers, guidance, history, disclaimer) | ✅ Delivered (proposal Week-11 item) |
| Deployment (HF Spaces) | Local deployment | ❌ Gap — local run documented; cloud deploy deferred to future work |

## 4. Evaluation metrics vs. targets

| Proposal target | Result | Verdict |
|---|---|---|
| ROC-AUC > 0.85 per disease | Diabetes 0.938 · Heart **0.801** · Liver 0.999 · Kidney 0.971 | 3/4 exceed; heart below target — see §6 |
| Multi-label (4 simultaneous) | 4 independent heads on unified schema | Delivered |
| Explainability: SHAP per patient | Verified per-instance, contributions sum to prediction | Delivered |
| API < 500 ms | ~7 ms predict, ~1 s report/LLM | Exceeded |
| Cross-disease interrelation | `interrelation_stress_test.csv`: Kidney→Heart +0.122, Liver→Diabetes +0.131, Liver→Heart +0.289 confirmed; Diabetes→Kidney/Heart not detected at dataset level | 3/5 — honest mixed result |
| Unit/integration tests | 19 pytest cases green | Delivered |

## 5. Delivered beyond the proposal (new contributions)

- **Cross-disease SHAP attribution** — per-patient drivers grouped by disease family
  (e.g., diabetes markers contribute +0.63 log-odds to a marker-less kidney score).
- **Panel ablation** — same patient scored full / own-markers-only / markers-removed,
  quantifying each panel's contribution vs. cross-disease transfer.
- **Marker-coverage gating** — `n/m markers` chips + confidence flags distinguish
  cohort-structural from patient-side missingness (a *finding* from testing: kidney
  ~95% on empty input exposed sentinel out-of-distribution extrapolation).
- **Dual-mode chat** — disease-scoped + overall-profile LLM chat with conversation
  memory and word-boundary disease routing.
- **Clinical context notes** — patient-facing lifestyle guidance from NLP extraction.

## 6. Deviations & their justification (for the report's Limitations/Discussion)

1. **Heart AUC 0.80 < 0.85 target.** The heart cohort is the weakest label set; FT-Transformer
   reached 0.813. Honest framing: metabolic diseases exceed target; cardiovascular risk
   needs richer features (ECG, family-history weighting) — future work.
2. **BioMistral/PubMedBERT → rule-based extractor.** A 7B clinical LLM for lifestyle
   extraction was disproportionate to the input (short patient-typed histories); the
   regex extractor covers the required factors with 10/10 test accuracy and zero
   latency/cost. PubMedBERT hook retained in code as the documented upgrade path.
3. **No MIMIC-III.** Access credentialing exceeded the semester timeline — the
   proposal's own risk table planned for this (fallback text source used instead).
4. **FT-Transformer experimental, not deployed.** Test AUCs (0.931 mean) marginally
   exceed XGBoost but artifacts came from inconsistent prior runs; XGBoost offers
   determinism, TreeSHAP interpretability, and <10 ms inference. Comparison kept as
   the DL deliverable — the proposal named DL an "enhancement, not requirement."
5. **No cloud deployment/RAG/multi-agent.** Semester-2 *extensions* scoped down to
   what validates the thesis: fusion + cross-disease analysis + explanations.

## 7. Test evidence (latest full battery)

- Engine: predict/readiness/coverage/attribution/ablation/SHAP — 14/14 checks
- API: 6 endpoints, validation edges, 7 ms — 12/12 checks
- NLP: 10/10 lifestyle phrasings, patient-facing context notes
- UI (agent-API driven): full predict flow, coverage chips, cross-disease section
- PDF: 6 pages, confidence boxes on marker-less diseases only
- LLM: live `llm_json` call verified (Groq gpt-oss-20b, ~1 s)
- pytest suite: 19/19
