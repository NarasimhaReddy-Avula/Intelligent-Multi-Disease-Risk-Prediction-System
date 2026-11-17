"""
Multi-Disease Risk Prediction — Streamlit Frontend (Semester 2 build)

Run with:
    .venv/bin/python -m streamlit run webapp/app.py

Deployed model: per-disease XGBoost classifiers (sentinel -999 missing encoding).
Explainability: TreeSHAP (pred_contribs) + optional online LLM guidance.
Outputs: risk scores, per-feature drivers, downloadable PDF patient report.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.inference import (
    Artifacts,
    DISEASE_ICONS,
    DISEASE_NAMES,
    DISEASE_PANELS,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    NORMAL_RANGES,
    RISK_COLORS,
    SENTINEL_MISSING_VALUE,
    TARGET_COLS,
    disease_readiness,
    format_raw_value,
    inverse_standardize_with_sentinel,
    cross_disease_attribution,
    load_artifacts,
    panel_ablation,
    panel_coverage,
    panel_hint,
    patient_to_xgb,
    predict,
    predict_proba_binary,
    provided_and_missing,
    risk_level,
    shap_contributions,
    sigmoid,
    structured_factors,
)
from src.explainability.shap_payload import build_structured_shap_factors
from src.llm.clients import load_openai_compatible_config_from_env
from src.llm.explain import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DISCLAIMER as LLM_DISCLAIMER,
    build_overall_payload_for_llm,
    build_patient_payload_for_llm,
    generate_ai_chat_reply,
    generate_ai_explanation,
)
from src.nlp.clinical_text_extractor import ClinicalTextExtractor, clinical_context_notes
from src.report import generate_patient_report

GLOBAL_EXPLAIN_MAX_ROWS = 2000

REFERENCE_DATASET_CANDIDATES = [
    PROJECT_ROOT / "datasets" / "combined_unified_sentinel_scaled.csv",
    PROJECT_ROOT / "datasets" / "combined_unified_masked_scaled.csv",
    PROJECT_ROOT / "datasets" / "combined_unified_sentinel.csv",
    PROJECT_ROOT / "datasets" / "combined_unified_masked.csv",
]

XGB_MODEL_FILES = {
    "diabetes_risk": "diabetes_XGBoost.pkl",
    "heart_disease_risk": "heart_XGBoost.pkl",
    "liver_disease_risk": "liver_XGBoost.pkl",
    "kidney_disease_risk": "kidney_XGBoost.pkl",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_artifacts() -> Artifacts:
    return load_artifacts()


@st.cache_resource
def _get_extractor() -> ClinicalTextExtractor:
    return ClinicalTextExtractor(use_bert=False)


def _try_import_shap():
    try:
        import shap
        import matplotlib.pyplot as plt
        return shap, plt, None
    except Exception as e:
        return None, None, e


def _patient_signature(patient_dict, feature_names) -> str:
    payload = {f: patient_dict.get(f) for f in feature_names}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _patient_raw_vector(patient_dict, feature_names) -> np.ndarray:
    vals = []
    for f in feature_names:
        v = patient_dict.get(f)
        try:
            vals.append(float(v) if v is not None else np.nan)
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals, dtype=np.float32)


def _display_names(feature_names):
    return [FEATURE_LABELS.get(f, f) for f in feature_names]


def _impact_adjective(share_pct: float) -> str:
    if share_pct >= 25.0:
        return "significantly"
    if share_pct >= 12.0:
        return "moderately"
    return "slightly"


def _feature_phrase(feat: str, raw_val) -> str:
    label = FEATURE_LABELS.get(feat, feat)
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return f"Missing {label}"
    if feat == "gender":
        return f"Gender ({format_raw_value(feat, raw_val)})"
    nr = NORMAL_RANGES.get(feat)
    if nr is None:
        return label
    lo, hi, _ = nr
    try:
        v = float(raw_val)
    except Exception:
        return label
    if v > hi:
        return f"High {label}"
    if v < lo:
        return f"Low {label}"
    return label


def _plain_english(disease_name, feature_names, shap_row, patient_dict, top_n=3):
    shap_row = np.asarray(shap_row, dtype=np.float32)
    abs_vals = np.abs(shap_row)
    total = float(abs_vals.sum()) or 1.0
    shares = abs_vals / total * 100.0
    order = np.argsort(abs_vals)[::-1]
    inc, dec = [], []
    for idx in order:
        feat = feature_names[int(idx)]
        contrib = float(shap_row[int(idx)])
        if contrib == 0.0:
            continue
        phrase = _feature_phrase(feat, patient_dict.get(feat))
        val_str = format_raw_value(feat, patient_dict.get(feat))
        adj = _impact_adjective(float(shares[int(idx)]))
        if contrib > 0 and len(inc) < top_n:
            inc.append(f"{phrase} ({val_str}) increased {disease_name} risk {adj}.")
        elif contrib < 0 and len(dec) < top_n:
            dec.append(f"{phrase} ({val_str}) decreased {disease_name} risk {adj}.")
        if len(inc) >= top_n and len(dec) >= top_n:
            break
    return inc, dec


def _plotly_diverging_bar(feature_names, shap_row, patient_dict, top_k: int, title: str):
    """Clean diverging horizontal bar of SHAP contributions (red=up, green=down)."""
    import plotly.graph_objects as go

    shap_row = np.asarray(shap_row, dtype=np.float32)
    order = np.argsort(np.abs(shap_row))[::-1][:top_k]
    order = order[np.argsort(shap_row[order])]

    labels, vals, colors, hover = [], [], [], []
    for i in order:
        feat = feature_names[int(i)]
        v = float(shap_row[int(i)])
        raw = patient_dict.get(feat)
        labels.append(f"{FEATURE_LABELS.get(feat, feat)}  ·  {format_raw_value(feat, raw)}")
        vals.append(v)
        colors.append("#ef4444" if v > 0 else "#22c55e")
        hover.append(f"{v:+.4f} log-odds")

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors, text=hover, textposition="outside",
        textfont=dict(size=10, color="#475569"),
    ))
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#0f172a")),
        xaxis_title="SHAP contribution (log-odds)   |   red = raises risk · green = lowers risk",
        xaxis=dict(title_font=dict(size=11, color="#64748b"), gridcolor="#e2e8f0", zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(320, 34 * len(vals) + 130),
        margin=dict(l=10, r=40, t=50, b=60),
        plot_bgcolor="white",
        bargap=0.35,
    )
    return fig


@st.cache_data
def _find_reference_dataset_path(numeric_cols_tuple):
    required = list(numeric_cols_tuple)
    for p in REFERENCE_DATASET_CANDIDATES:
        if not p.exists():
            continue
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
            if all(c in cols for c in required):
                return str(p)
        except Exception:
            continue
    return None


@st.cache_data
def _global_shap_bundle(model_sigs, dataset_sig, numeric_cols_tuple, max_rows):
    import xgboost as xgb
    import joblib

    df = pd.read_csv(dataset_sig[0], usecols=list(numeric_cols_tuple), nrows=int(max_rows))
    X_ref = df.to_numpy(dtype=np.float32)
    dmat = xgb.DMatrix(X_ref, feature_names=list(numeric_cols_tuple))
    bundle = {}
    for target, (mp, _mt, _sz) in model_sigs.items():
        model = joblib.load(mp)
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        contribs = np.asarray(booster.predict(dmat, pred_contribs=True))
        bundle[target] = {"shap_vals": contribs[:, :-1].astype(np.float32), "bias": contribs[:, -1].astype(np.float32)}
    return X_ref, bundle


def _mentioned_disease_targets(text: str):
    import re
    t = (text or "").lower()
    def has_any(words):          # substring ok for unambiguous disease names
        return any(w in t for w in words)
    def has_word(words):         # whole-word only — "alt" would match "salt"/"health" otherwise
        return any(re.search(rf"\b{re.escape(w)}\b", t) for w in words)
    explicit = {
        "kidney_disease_risk": ["kidney", "renal", "nephro"],
        "liver_disease_risk": ["liver", "hepatic", "hepat"],
        "heart_disease_risk": ["heart", "cardio", "cardiac", "cardiovascular", "coronary"],
        "diabetes_risk": ["diabetes", "diabetic"],
    }
    hits = [tg for tg, ws in explicit.items() if has_any(ws)]
    if len(hits) == 1:
        return hits
    if has_word(["gfr", "egfr", "creatinine", "bun", "proteinuria"]) or "urine protein" in t:
        hits.append("kidney_disease_risk")
    if has_word(["alt", "ast", "alp", "bilirubin", "sgpt", "sgot"]) or "jaundice" in t:
        hits.append("liver_disease_risk")
    if has_word(["hba1c", "a1c", "glucose"]) or has_any(["fasting glucose", "blood sugar", "sugar level"]):
        hits.append("diabetes_risk")
    if has_word(["ldl", "hdl", "cholesterol", "triglycerides", "triglyceride"]):
        hits.append("heart_disease_risk")
    uniq = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    return uniq


_OVERALL_WORDS = ("overall", "all of", "everything", "both", "general", "combined",
                  "full picture", "whole", "every disease", "four", "all four")


def _wants_overall(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in _OVERALL_WORDS)


def _lifestyle_context_dict(lf) -> dict:
    """Flatten LifestyleFeatures into a compact dict for the LLM payload/PDF."""
    if lf is None:
        return {}
    out = {
        "smoking_status": lf.smoking_status,
        "smoking_frequency": lf.smoking_frequency,
        "alcohol_status": lf.alcohol_status,
        "alcohol_frequency": lf.alcohol_frequency,
        "physical_activity": lf.physical_activity,
        "family_history": list(lf.family_history or []),
        "symptoms": list(lf.symptoms or []),
    }
    return {k: v for k, v in out.items() if v not in (None, "unknown", [])}


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedRisk AI — Multi-Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c75 100%);
    border-radius: 16px; padding: 2.2rem 2.6rem; margin-bottom: 1.6rem;
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 { font-family: 'DM Serif Display', serif; color: #f0f9ff;
    font-size: 2.3rem; margin: 0; letter-spacing: -0.5px; }
.main-header p { color: #7dd3fc; margin: 0.4rem 0 0 0; font-size: 1rem; font-weight: 300; }
.badge { display: inline-block; background: rgba(56,189,248,0.2);
    border: 1px solid rgba(56,189,248,0.4); color: #7dd3fc;
    padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem;
    margin-top: 0.8rem; font-weight: 500; letter-spacing: 0.5px; }
.risk-card { border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    border: 1px solid; }
.risk-card-title { font-size: 0.82rem; font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; opacity: 0.7; margin-bottom: 0.3rem; }
.risk-card-disease { font-family: 'DM Serif Display', serif; font-size: 1.25rem;
    font-weight: 400; margin-bottom: 0.5rem; }
.risk-card-prob { font-size: 2.6rem; font-weight: 600; line-height: 1; margin-bottom: 0.3rem; }
.risk-card-level { font-size: 0.8rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
.prob-bar-container { background: rgba(0,0,0,0.08); border-radius: 4px; height: 6px;
    margin: 0.5rem 0; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.section-header { font-family: 'DM Serif Display', serif; font-size: 1.25rem;
    color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem;
    margin-bottom: 1.1rem; margin-top: 1.4rem; }
.missing-box { background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px;
    padding: 1rem 1.2rem; margin-top: 1rem; font-size: 0.85rem; color: #64748b; }
.warning-banner { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 10px;
    padding: 0.8rem 1.2rem; font-size: 0.85rem; color: #92400e; margin-top: 1rem; }
.lifestyle-chip { display: inline-block; background: #eff6ff; border: 1px solid #bfdbfe;
    color: #1e40af; padding: 0.25rem 0.7rem; border-radius: 14px; font-size: 0.8rem;
    margin: 0.15rem 0.25rem 0.15rem 0; }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox input {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background-color: #f8fafc !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
    color: #0f172a !important; }
section[data-testid="stSidebar"] .stNumberInput button {
    background-color: #e2e8f0 !important; }
div.stButton > button { background: linear-gradient(135deg, #0369a1, #0284c7);
    color: white; border: none; border-radius: 10px; padding: 0.7rem 2.2rem;
    font-size: 1rem; font-weight: 600; width: 100%; }
div.stButton > button:hover { background: linear-gradient(135deg, #0284c7, #0ea5e9);
    transform: translateY(-1px); box-shadow: 0 4px 15px rgba(3,105,161,0.4); }
div.stDownloadButton > button { background: linear-gradient(135deg, #0f766e, #14b8a6);
    color: white; border: none; border-radius: 10px; padding: 0.7rem 2.2rem;
    font-size: 1rem; font-weight: 600; width: 100%; }
.normal-tag { font-size: 0.72rem; color: #94a3b8; margin-top: -0.5rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    artifacts = _load_artifacts()
    xgb_models = artifacts.models
    scaler_bundle = artifacts.scaler_bundle
    numeric_cols = artifacts.numeric_cols
    model_loaded = True
    model_auc = artifacts.mean_auc
except Exception as e:
    model_loaded = False
    model_error = str(e)
    artifacts = None
    xgb_models = {}
    scaler_bundle = None
    numeric_cols = list(FEATURE_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏥 MedRisk AI</h1>
    <p>Intelligent Multi-Disease Risk Prediction Platform</p>
    <span class="badge">XGBoost · SHAP Explanations · PDF Reports · IIIT Sri City BTP 2025–26</span>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model could not be loaded: {model_error}")
    st.info("Expected artifacts: week7_artifacts/scaler.pkl and week7_artifacts/{diabetes,heart,liver,kidney}_XGBoost.pkl")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Model", "XGBoost")
c2.metric("Mean AUC", f"{model_auc:.4f}" if model_auc is not None else "n/a")
c3.metric("Diseases", "4 risk models")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PATIENT INFO
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 👤 Patient Information")
    patient_name = st.text_input("Patient Name", value="", placeholder="e.g. Rahul Sharma")
    patient_age = st.number_input("Age (quick entry)", min_value=1, max_value=120, value=45)
    patient_gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "- Select the lab tests available and enter values\n"
        "- Missing tests are handled automatically (sentinel encoding)\n"
        "- Optionally paste a patient history — lifestyle factors are extracted and used to enrich explanations\n"
        "- Download a full PDF report for all 4 diseases\n\n"
        "**Risk levels:** 🟢 LOW <35% · 🟡 MEDIUM 35–65% · 🔴 HIGH >65%"
    )
    if artifacts is not None and artifacts.aucs:
        st.markdown("---")
        st.markdown("### 📈 Model performance (test AUC)")
        for t in TARGET_COLS:
            nm = DISEASE_NAMES[TARGET_COLS.index(t)]
            st.markdown(f"- {nm}: **{artifacts.aucs[t]:.3f}**")
        st.caption("XGBoost, per-disease heads · unified 24-feature schema")
    st.markdown("---")
    st.caption("⚠️ For clinical research only. Not a substitute for medical advice.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 🧪 Enter Lab Values")
st.markdown("Select available tests and enter values. Unselected fields are auto-handled by the model.")

patient_dict = {"gender": 1.0 if patient_gender == "Male" else 0.0, "age": float(patient_age)}

for group_name, features in FEATURE_GROUPS.items():
    if group_name == "Demographics":
        with st.expander(f"👤 {group_name} — age & gender set in sidebar", expanded=False):
            st.info(f"Age: {patient_age} | Gender: {patient_gender}")
            enter_bmi = st.checkbox("Enter BMI", value=True, key="cb_bmi")
            if enter_bmi:
                bmi_val = st.number_input(FEATURE_LABELS["bmi"], min_value=10.0, max_value=80.0,
                                          value=25.0, step=0.1, key="bmi")
                st.markdown('<div class="normal-tag">Normal: 18.5 – 24.9 kg/m²</div>', unsafe_allow_html=True)
                patient_dict["bmi"] = bmi_val
            else:
                patient_dict["bmi"] = None
        continue

    icon = {"Cardiovascular": "🫀", "Diabetes Markers": "🩸", "Liver Function": "🫁", "Kidney Function": "🫘"}.get(group_name, "🧪")
    with st.expander(f"{icon} {group_name}", expanded=True):
        cols = st.columns(2)
        for idx, feat in enumerate(features):
            with cols[idx % 2]:
                enter = st.checkbox(f"Enter {FEATURE_LABELS[feat]}", value=False, key=f"cb_{feat}")
                if enter:
                    nr = NORMAL_RANGES.get(feat)
                    if nr:
                        lo, hi, _ = nr
                        default_v = float((lo + hi) / 2)
                        max_v = max(float(hi) * 5.0, default_v + 1.0, 10.0)
                    else:
                        lo, hi, default_v, max_v = 0.0, 0.0, 0.0, 1000.0
                    val = st.number_input(FEATURE_LABELS[feat], min_value=0.0, max_value=float(max_v),
                                          value=default_v, step=0.1, key=f"val_{feat}",
                                          label_visibility="collapsed")
                    if nr:
                        flag = "✅" if lo <= val <= hi else "⚠️ Outside normal"
                        st.markdown(f'<div class="normal-tag">Normal: {lo}–{hi} {nr[2]} {flag}</div>',
                                    unsafe_allow_html=True)
                    patient_dict[feat] = val
                else:
                    patient_dict[feat] = None


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT HISTORY (NLP) — enriches explanations, not model inputs
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 📝 Patient History (optional — enriches explanations)")
st.caption(
    "Free-text lifestyle & family history, e.g. *\"I smoke 10 cigarettes daily, "
    "drink on weekends, sedentary job. Father had diabetes.\"* "
    "Extracted factors are shown in explanations and the PDF report — they do **not** affect your risk scores."
)

history_text = st.text_area("Patient history text", height=90, label_visibility="collapsed",
                            placeholder="Describe lifestyle habits, family history, symptoms...")

lifestyle_features = None
lifestyle_ctx = {}
if history_text and history_text.strip():
    extractor = _get_extractor()
    lifestyle_features = extractor.extract_features(history_text.strip())
    lifestyle_ctx = _lifestyle_context_dict(lifestyle_features)
    if lifestyle_ctx:
        chips = []
        label_map = {
            "smoking_status": "🚬 smoking", "smoking_frequency": "amount",
            "alcohol_status": "🍺 alcohol", "alcohol_frequency": "amount",
            "physical_activity": "🏃 activity", "family_history": "👪 family hx",
            "symptoms": "🩺 symptoms",
        }
        for k, v in lifestyle_ctx.items():
            vv = ", ".join(v) if isinstance(v, list) else str(v)
            chips.append(f'<span class="lifestyle-chip">{label_map.get(k, k)}: {vv}</span>')
        st.markdown(" ".join(chips), unsafe_allow_html=True)

        ctx_notes = clinical_context_notes(lifestyle_features)
        if ctx_notes:
            for n in ctx_notes:
                st.markdown(
                    f'<div class="warning-banner" style="margin-top:0.4rem;">'
                    f'⚠️ <strong>{n["factor"]}</strong> — {n["note"]}</div>',
                    unsafe_allow_html=True)
            st.caption("General health notes based on your reported lifestyle.")
    else:
        st.caption("No lifestyle factors detected in the text.")


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDED / MISSING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

provided, missing = provided_and_missing(patient_dict, numeric_cols)
st.markdown("---")
col_a, col_b = st.columns(2)
col_a.markdown(f"**✅ Provided:** {len(provided)} / {len(numeric_cols)} features")
if provided:
    col_a.caption(", ".join(FEATURE_LABELS.get(f, f) for f in provided))
col_b.markdown(f"**⚪ Missing (auto-handled):** {len(missing)} features")
if missing:
    col_b.caption(", ".join(FEATURE_LABELS.get(f, f) for f in missing))

if len(provided) < 3:
    st.markdown('<div class="warning-banner">⚠️ Very few features provided — predictions will rely heavily on '
                'demographics. Enter a few lab values for meaningful results.</div>', unsafe_allow_html=True)
st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [("pred_results", None), ("pred_patient", None), ("pred_name", None),
                     ("pred_history", None), ("pred_lifestyle", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.button("🔍 Predict Disease Risk", type="primary"):
    with st.spinner("Running XGBoost inference..."):
        try:
            results = predict(patient_dict, artifacts)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    st.session_state["pred_results"] = results
    st.session_state["pred_patient"] = dict(patient_dict)
    st.session_state["pred_name"] = patient_name.strip() or "Patient"
    st.session_state["pred_history"] = history_text.strip() if history_text else None
    st.session_state["pred_lifestyle"] = lifestyle_features

if st.session_state.get("pred_results") is None:
    st.stop()

results = st.session_state["pred_results"]
patient_used = st.session_state.get("pred_patient") or dict(patient_dict)
pname = st.session_state.get("pred_name") or "Patient"
history_used = st.session_state.get("pred_history")
lifestyle_used = st.session_state.get("pred_lifestyle")
provided_used, missing_used = provided_and_missing(patient_used, numeric_cols)

st.markdown("---")
st.markdown(f"## 📊 Risk Assessment — {pname}")
st.markdown(f"*{len(provided_used)} features provided · {len(missing_used)} auto-handled*")

# ── Risk cards ────────────────────────────────────────────────────────────────
# Per-disease readiness: a model score is only meaningful if the patient provided
# at least one marker from that disease's lab panel (models were trained on
# cohort-specific rows — unseen sentinels extrapolate unreliably).
readiness = disease_readiness(patient_used)
coverage = panel_coverage(patient_used)

cols = st.columns(4)
for i, (target, name, icon) in enumerate(zip(TARGET_COLS, DISEASE_NAMES, DISEASE_ICONS)):
    rdy = readiness[target]
    prob = results[target]
    level = risk_level(prob)
    color, bg = RISK_COLORS[level]
    with cols[i]:
        st.markdown(f"""
        <div class="risk-card" style="background:{bg}; border-color:{color}40; color:#0f172a;">
            <div class="risk-card-title">{icon} Disease Risk</div>
            <div class="risk-card-disease">{name}</div>
            <div class="risk-card-prob" style="color:{color};">{prob*100:.1f}%</div>
            <div class="prob-bar-container"><div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:{color};"></div></div>
            <div class="risk-card-level" style="color:{color};">{level} RISK</div>
        </div>""", unsafe_allow_html=True)
        cov = coverage[target]
        bar_pct = int(cov["pct"] * 100)
        st.markdown(
            f'<div style="margin-top:0.35rem;"><div style="font-size:0.72rem;color:#64748b;">'
            f'{cov["provided"]}/{cov["total"]} own markers</div>'
            f'<div style="background:#e2e8f0;border-radius:4px;height:5px;">'
            f'<div style="width:{bar_pct}%;background:#0284c7;height:100%;border-radius:4px;"></div>'
            f'</div></div>', unsafe_allow_html=True)
        if not rdy["scoreable"]:
            st.caption(
                f"⚠️ Based on cross-disease features only — for a more accurate score, "
                f"provide at least one: {panel_hint(target)}"
            )

# ── PDF REPORT DOWNLOAD ──────────────────────────────────────────────────────
st.markdown("")
with st.expander("📄 Patient Report (PDF)", expanded=True):
    st.markdown(
        "Downloads a **single multi-disease PDF report**: predictions for all 4 diseases, "
        "per-disease reasoning (SHAP drivers), cautions & red flags, doctors to consult, "
        "diet/lifestyle guidance, and patient-reported history."
    )
    try:
        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_patient_report(
                patient_dict=patient_used,
                results=results,
                artifacts=artifacts,
                patient_name=pname,
                history_text=history_used,
                lifestyle_features=lifestyle_used,
            )
        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"medrisk_report_{(pname or 'patient').replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
        )
        st.caption("6 pages: summary + per-disease assessment + methodology & disclosures.")
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")

# ── EXPLAINABILITY (local) ───────────────────────────────────────────────────
scaler = scaler_bundle["scaler"]
scaler_cols = list(scaler_bundle["numeric_cols"])
X = patient_to_xgb(patient_used, numeric_cols, scaler, scaler_cols)

local_cache_key = f"{_patient_signature(patient_used, numeric_cols)}::{artifacts.artifact_dir}"
if st.session_state.get("local_shap_cache_key") != local_cache_key:
    st.session_state["local_shap_cache_key"] = local_cache_key
    st.session_state["local_shap_cache"] = {}
local_cache = st.session_state["local_shap_cache"]

def _get_shap_row(target):
    if target in local_cache:
        return (np.asarray(local_cache[target]["shap_row"], dtype=np.float32),
                float(local_cache[target]["bias_val"]))
    sv, b = shap_contributions(xgb_models[target], X, numeric_cols)
    row = np.asarray(sv[0], dtype=np.float32)
    local_cache[target] = {"shap_row": row, "bias_val": float(b[0])}
    st.session_state["local_shap_cache"] = local_cache
    return row, float(b[0])


with st.expander("🔎 Explainability — What drives each risk score", expanded=True):
    st.markdown(
        "Which inputs pushed each disease risk **up** (red) or **down** (green). "
        "Contributions are TreeSHAP values in **log-odds** space — they add up to the prediction."
    )
    max_k = min(20, len(numeric_cols))
    top_k = st.slider("Top features to show", min_value=3, max_value=max_k,
                      value=min(10, max_k), key="exp_top_k") if max_k >= 3 else max_k

    tabs = st.tabs([f"{icon} {name}" for icon, name in zip(DISEASE_ICONS, DISEASE_NAMES)])
    for tab, target, disease_name in zip(tabs, TARGET_COLS, DISEASE_NAMES):
        with tab:
            if not readiness[target]["scoreable"]:
                st.caption(
                    f"⚠️ Cross-disease estimate only — no {disease_name.lower()}-specific markers "
                    f"provided. For better accuracy, enter at least one: {panel_hint(target)}"
                )
            try:
                shap_row, bias_val = _get_shap_row(target)

                fig = _plotly_diverging_bar(
                    list(numeric_cols), shap_row, patient_used, int(top_k),
                    title=f"{disease_name}: top feature drivers",
                )
                st.plotly_chart(fig, use_container_width=True)

                margin = bias_val + float(np.sum(shap_row))
                st.caption(
                    f"Base risk≈{sigmoid(bias_val)*100:.1f}% · "
                    f"Explained risk≈{sigmoid(margin)*100:.1f}% · "
                    f"Model risk={results[target]*100:.1f}%"
                )

                inc, dec = _plain_english(disease_name, list(numeric_cols), shap_row, patient_used, top_n=3)
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**⬆️ Factors increasing risk**")
                    st.markdown("\n".join(f"- {s}" for s in inc) if inc else "_None dominant._")
                with col_t2:
                    st.markdown("**⬇️ Factors decreasing risk**")
                    st.markdown("\n".join(f"- {s}" for s in dec) if dec else "_None dominant._")

                with st.expander("Detailed view (SHAP waterfall + force + raw table)", expanded=False):
                    shap, plt, shap_err = _try_import_shap()
                    if shap is None:
                        st.info("SHAP plots need `shap` + `matplotlib` in this environment.")
                    else:
                        try:
                            raw_vec = _patient_raw_vector(patient_used, numeric_cols)
                            exp = shap.Explanation(values=shap_row, base_values=bias_val,
                                                   data=raw_vec, feature_names=_display_names(numeric_cols))
                            c_p1, c_p2 = st.columns(2)
                            with c_p1:
                                plt.figure(); shap.plots.waterfall(exp, max_display=int(top_k), show=False)
                                st.pyplot(plt.gcf(), use_container_width=True, clear_figure=True)
                            with c_p2:
                                plt.figure(); shap.plots.bar(exp, max_display=int(top_k), show=False)
                                st.pyplot(plt.gcf(), use_container_width=True, clear_figure=True)
                            force = shap.force_plot(float(bias_val), shap_row, raw_vec,
                                                    feature_names=_display_names(numeric_cols), matplotlib=False)
                            components.html(f"<head>{shap.getjs()}</head><body>{force.html()}</body>",
                                            height=280, scrolling=True)
                        except Exception as e:
                            st.warning(f"Could not render SHAP plots: {e}")

                    rows = []
                    abs_v = np.abs(shap_row)
                    denom = float(abs_v.sum()) or 1.0
                    for feat, contrib in zip(numeric_cols, shap_row):
                        raw_val = patient_used.get(feat)
                        rows.append({
                            "Feature": FEATURE_LABELS.get(feat, feat),
                            "Raw value": format_raw_value(feat, raw_val),
                            "Contribution (log-odds)": round(float(contrib), 4),
                            "Impact share (%)": round(float(abs(contrib)) / denom * 100.0, 1),
                        })
                    st.dataframe(pd.DataFrame(rows).sort_values("Impact share (%)", ascending=False).head(int(top_k)),
                                 use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Explainability unavailable for {disease_name}: {e}")

# ── EXPLAINABILITY (global) ──────────────────────────────────────────────────
with st.expander("🌍 Explainability — Global Feature Importance", expanded=False):
    st.markdown("Which features matter **overall** for each disease model — mean |SHAP| on a reference sample.")
    shap, plt, shap_err = _try_import_shap()
    if shap is None:
        st.info("Global SHAP needs `shap` + `matplotlib` in this environment.")
    else:
        ref_path = _find_reference_dataset_path(tuple(numeric_cols))
        if ref_path is None:
            st.warning("No reference dataset found (combined_unified_*_scaled.csv under datasets/).")
        else:
            ref_p = Path(ref_path)
            ds_stat = ref_p.stat()
            dataset_sig = (str(ref_p), float(ds_stat.st_mtime), int(ds_stat.st_size))
            artifact_dir_path = Path(artifacts.artifact_dir)
            model_sigs = {}
            for target, fname in XGB_MODEL_FILES.items():
                mp = artifact_dir_path / fname
                if mp.exists():
                    st_mp = mp.stat()
                    model_sigs[target] = (str(mp), float(st_mp.st_mtime), int(st_mp.st_size))
            if model_sigs:
                try:
                    X_ref, global_bundle = _global_shap_bundle(
                        model_sigs, dataset_sig, tuple(numeric_cols), GLOBAL_EXPLAIN_MAX_ROWS)
                    st.caption(f"Reference: {ref_p.name} · {X_ref.shape[0]} rows · {len(numeric_cols)} features")
                    X_ref_raw = inverse_standardize_with_sentinel(X_ref, scaler, SENTINEL_MISSING_VALUE)
                    display_names = _display_names(numeric_cols)
                    tabs_g = st.tabs([f"{i} {n}" for i, n in zip(DISEASE_ICONS, DISEASE_NAMES)])
                    for tab, target, disease_name in zip(tabs_g, TARGET_COLS, DISEASE_NAMES):
                        with tab:
                            if target not in global_bundle:
                                continue
                            shap_vals = global_bundle[target]["shap_vals"]
                            exp_g = shap.Explanation(values=shap_vals,
                                                     base_values=np.asarray(global_bundle[target]["bias"]),
                                                     data=X_ref_raw, feature_names=display_names)
                            plt.figure(); shap.plots.bar(exp_g, max_display=15, show=False)
                            st.pyplot(plt.gcf(), use_container_width=True, clear_figure=True)
                            plt.figure(); shap.plots.beeswarm(exp_g, max_display=15, show=False)
                            st.pyplot(plt.gcf(), use_container_width=True, clear_figure=True)
                except Exception as e:
                    st.warning(f"Global explainability failed: {e}")

# ── AI EXPLANATION & CHAT (online LLM) ───────────────────────────────────────
with st.expander("🤖 AI Explanation & Guidance (Online LLM)", expanded=False):
    st.markdown("Patient-friendly explanation and Q&A via an **online** LLM (OpenAI-compatible). "
                "Keys come from environment variables.")
    cfg = load_openai_compatible_config_from_env(default_base_url=DEFAULT_BASE_URL, default_model=DEFAULT_MODEL)
    if cfg is None:
        st.info("LLM not configured. Set `BTP_LLM_API_KEY` (or `GROQ_API_KEY` for Groq) and restart. "
                "A rule-based fallback is used meanwhile.")
    else:
        st.success(f"LLM configured: {cfg.base_url} · model={cfg.model}")

    ai_mode = st.radio(
        "Explanation scope",
        ["🧑‍⚕️ Overall assessment (all 4 diseases)", "🎯 Single disease"],
        horizontal=True, key="ai_mode",
    )

    def _disease_llm_entry(target: str) -> Dict[str, Any]:
        row, _b = _get_shap_row(target)
        fac = build_structured_shap_factors(
            feature_names=list(numeric_cols), shap_row=row, patient_dict=patient_used,
            feature_labels=FEATURE_LABELS, normal_ranges=NORMAL_RANGES, top_k=6)
        return {
            "name": DISEASE_NAMES[TARGET_COLS.index(target)],
            "target": target,
            "probability": float(results[target]),
            "level": risk_level(float(results[target])),
            "insufficient_markers": not readiness[target]["scoreable"],
            "top_increasing": fac["top_increasing"],
            "top_decreasing": fac["top_decreasing"],
        }

    if ai_mode.startswith("🎯"):
        ai_target = st.selectbox("Disease model to explain", options=TARGET_COLS,
                                 format_func=lambda t: f"{DISEASE_ICONS[TARGET_COLS.index(t)]} {DISEASE_NAMES[TARGET_COLS.index(t)]}",
                                 key="ai_target")
        entry = _disease_llm_entry(ai_target)
        payload_ai = build_patient_payload_for_llm(
            disease_name=entry["name"], probability=entry["probability"], risk_level=entry["level"],
            top_increasing=entry["top_increasing"], top_decreasing=entry["top_decreasing"],
            missing_features=[FEATURE_LABELS.get(f, f) for f in missing_used])
        if lifestyle_ctx:
            payload_ai["patient_reported"] = lifestyle_ctx
        ai_cache_key = f"{_patient_signature(patient_used, numeric_cols)}::{ai_target}::{getattr(cfg, 'model', 'rules')}::{hashlib.sha1((history_used or '').encode()).hexdigest()[:8]}"
    else:
        diseases_payload = [_disease_llm_entry(t) for t in TARGET_COLS]
        payload_ai = build_overall_payload_for_llm(
            diseases=diseases_payload,
            missing_features=[FEATURE_LABELS.get(f, f) for f in missing_used],
            patient_reported=lifestyle_ctx)
        ai_cache_key = f"{_patient_signature(patient_used, numeric_cols)}::overall::{getattr(cfg, 'model', 'rules')}::{hashlib.sha1((history_used or '').encode()).hexdigest()[:8]}"

    if "ai_explain_cache" not in st.session_state:
        st.session_state["ai_explain_cache"] = {}
    if st.button("Generate AI explanation", key="ai_generate"):
        with st.spinner("Generating explanation..."):
            st.session_state["ai_explain_cache"][ai_cache_key] = generate_ai_explanation(payload=payload_ai, cfg=cfg)

    out = st.session_state["ai_explain_cache"].get(ai_cache_key)
    if out is not None:
        if not out.get("ok", True):
            st.warning(f"LLM call issue: {out.get('error', 'unknown')} (showing fallback)")
        else:
            _src = {"llm_json": "AI-generated", "llm_text": "AI-generated",
                    "rules": "Rule-based (no LLM key)"}.get(out.get("mode"), out.get("mode", ""))
            if _src:
                st.caption(f"Source: {_src}" + (f" · {out['latency_s']:.1f}s" if out.get("latency_s") else ""))
        data = out.get("data") or {}
        if data.get("summary"):
            st.markdown("### Summary")
            st.write(data["summary"])
        inc_list, dec_list = data.get("key_factors_increasing"), data.get("key_factors_decreasing")
        if inc_list or dec_list:
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Key factors increasing risk**")
                st.markdown("\n".join(f"- {s}" for s in inc_list) if isinstance(inc_list, list) and inc_list else "_n/a_")
            with cc2:
                st.markdown("**Key factors decreasing risk**")
                st.markdown("\n".join(f"- {s}" for s in dec_list) if isinstance(dec_list, list) and dec_list else "_n/a_")
        if isinstance(data.get("what_to_do_next"), list) and data["what_to_do_next"]:
            st.markdown("### What to do next")
            st.markdown("\n".join(f"- {s}" for s in data["what_to_do_next"]))
        if isinstance(data.get("when_to_seek_care"), list) and data["when_to_seek_care"]:
            st.markdown("### When to seek care")
            st.markdown("\n".join(f"- {s}" for s in data["when_to_seek_care"]))
        st.caption(f"⚠️ {data.get('disclaimer') or LLM_DISCLAIMER}")

    st.markdown("---")
    st.markdown("### 💬 Ask follow-up questions")
    if st.session_state.get("ai_chat_cache_key") != ai_cache_key:
        st.session_state["ai_chat_cache_key"] = ai_cache_key
        st.session_state["ai_chat_history"] = []
        st.session_state["ai_last_target"] = None
    for msg in st.session_state.get("ai_chat_history", []):
        if msg.get("content"):
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg["content"])

    user_q = st.chat_input("Ask about kidney/liver/heart/diabetes risk and the shown factors...")
    if user_q:
        st.session_state["ai_chat_history"] = st.session_state["ai_chat_history"] + [{"role": "user", "content": user_q}]
        _chat_hist = st.session_state["ai_chat_history"][:-1]  # exclude just-added message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                mentioned = _mentioned_disease_targets(user_q)
                if len(mentioned) > 1:
                    names = [DISEASE_NAMES[TARGET_COLS.index(t)] for t in mentioned if t in TARGET_COLS]
                    answer = ("Your question mentions multiple diseases (" + ", ".join(names) + "). "
                              "Please ask about one disease at a time. Example: 'What drives my kidney risk?'")
                    reply = {"answer": answer, "disclaimer": LLM_DISCLAIMER}
                else:
                    sticky = False
                    if len(mentioned) == 1:
                        chat_target = mentioned[0]
                        st.session_state["ai_last_target"] = chat_target
                    elif _wants_overall(user_q):
                        chat_target = None
                    else:
                        # No disease named → follow the conversation's current disease, else overall
                        chat_target = st.session_state.get("ai_last_target")
                        sticky = chat_target is not None
                    if chat_target is None:
                        chat_payload = build_overall_payload_for_llm(
                            diseases=[_disease_llm_entry(t) for t in TARGET_COLS],
                            missing_features=[FEATURE_LABELS.get(f, f) for f in missing_used],
                            patient_reported=lifestyle_ctx)
                        reply = generate_ai_chat_reply(payload=chat_payload, user_message=user_q,
                                                       chat_history=_chat_hist, cfg=cfg)
                        answer = "*Context used: overall 4-disease profile*\n\n" + (reply.get("answer") or "(no answer)")
                    else:
                        chat_disease = DISEASE_NAMES[TARGET_COLS.index(chat_target)]
                        chat_entry = _disease_llm_entry(chat_target)
                        chat_payload = build_patient_payload_for_llm(
                            disease_name=chat_entry["name"], probability=chat_entry["probability"],
                            risk_level=chat_entry["level"],
                            top_increasing=chat_entry["top_increasing"],
                            top_decreasing=chat_entry["top_decreasing"],
                            missing_features=[FEATURE_LABELS.get(f, f) for f in missing_used])
                        if lifestyle_ctx:
                            chat_payload["patient_reported"] = lifestyle_ctx
                        reply = generate_ai_chat_reply(payload=chat_payload, user_message=user_q,
                                                       chat_history=_chat_hist, cfg=cfg)
                        tag = " (follow-up)" if sticky else ""
                        answer = f"*Context used: {chat_disease} model{tag}*\n\n" + (reply.get("answer") or "(no answer)")
                st.markdown(answer)
                st.caption(f"⚠️ {reply.get('disclaimer') or LLM_DISCLAIMER}")
        st.session_state["ai_chat_history"] = st.session_state["ai_chat_history"] + [
            {"role": "assistant", "content": answer}]

# ── DETAILED TABLE + CROSS-DISEASE INSIGHTS ──────────────────────────────────
with st.expander("📋 Detailed Results Table", expanded=False):
    st.dataframe(pd.DataFrame([{
        "Disease": f"{icon} {name}",
        "Probability": f"{results[t]:.4f}",
        "Risk (%)": f"{results[t]*100:.2f}%",
        "Risk Level": risk_level(results[t]),
        "Own markers": "✓" if readiness[t]["scoreable"] else "none — lower confidence",
    } for t, name, icon in zip(TARGET_COLS, DISEASE_NAMES, DISEASE_ICONS)]),
        use_container_width=True, hide_index=True)

st.markdown("### 🔗 Cross-Disease Insights")
st.caption("Clinical heuristics based on the 4 risk scores (models are trained per-disease).")
d_risk, h_risk, l_risk, k_risk = (results["diabetes_risk"], results["heart_disease_risk"],
                                  results["liver_disease_risk"], results["kidney_disease_risk"])
insights = []
if d_risk >= 0.5 and k_risk >= 0.35:
    insights.append(("🩸→🫘 Diabetic Nephropathy", f"Diabetes {d_risk*100:.0f}% + kidney {k_risk*100:.0f}% suggests possible diabetic nephropathy. Monitor kidney function."))
if d_risk >= 0.5 and h_risk >= 0.35:
    insights.append(("🩸→❤️ Diabetic Cardiomyopathy", f"Diabetes {d_risk*100:.0f}% + heart {h_risk*100:.0f}% may indicate diabetic cardiomyopathy."))
if k_risk >= 0.5 and h_risk >= 0.35:
    insights.append(("🫘→❤️ Cardiorenal Syndrome", f"Kidney {k_risk*100:.0f}% + heart {h_risk*100:.0f}% — these diseases often co-occur."))
if l_risk >= 0.5 and d_risk >= 0.35:
    insights.append(("🫁→🩸 NAFLD–Insulin Resistance", f"Liver {l_risk*100:.0f}% + diabetes {d_risk*100:.0f}% may indicate NAFLD with insulin resistance."))
if l_risk >= 0.5 and h_risk >= 0.35:
    insights.append(("🫁→❤️ NAFLD–Cardiovascular", f"Liver {l_risk*100:.0f}% + heart {h_risk*100:.0f}% is associated with NAFLD-related CV risk."))

if insights:
    for title, desc in insights:
        st.markdown(f"""<div style="background:#f0f9ff; border-left:4px solid #0369a1;
            border-radius:0 10px 10px 0; padding:0.9rem 1.1rem; margin-bottom:0.7rem;">
            <strong style="color:#0369a1;">{title}</strong><br>
            <span style="color:#334155; font-size:0.9rem;">{desc}</span></div>""",
            unsafe_allow_html=True)
else:
    st.markdown("""<div style="background:#f0fdf4; border-left:4px solid #22c55e;
        border-radius:0 10px 10px 0; padding:0.9rem 1.1rem;">
        <strong style="color:#15803d;">✅ No significant cross-disease patterns detected.</strong><br>
        <span style="color:#334155; font-size:0.9rem;">Continue routine monitoring.</span></div>""",
        unsafe_allow_html=True)

# ── CROSS-DISEASE SIGNAL ANALYSIS ────────────────────────────────────────────
with st.expander("🔬 Cross-Disease Signal — who drives each score?", expanded=False):
    st.markdown(
        "Each score's SHAP drivers grouped by **which disease family the features belong to** — "
        "plus an ablation: how the score changes when a disease's own markers are removed. "
        "This quantifies how much fused, cross-disease data contributes to each prediction."
    )
    try:
        import plotly.graph_objects as go
        cd_rows, abl_rows = [], []
        attr_by_target = {}
        for t in TARGET_COLS:
            attr = cross_disease_attribution(t, patient_used, artifacts)
            attr_by_target[t] = attr
            abl = panel_ablation(t, patient_used, artifacts)
            abl_rows.append({
                "Disease": DISEASE_NAMES[TARGET_COLS.index(t)],
                "All features": f"{abl['full']*100:.1f}%",
                "Own markers only": f"{abl['panel_only']*100:.1f}%",
                "Own markers removed": f"{abl['panel_removed']*100:.1f}%",
                "Cross-disease share": f"{(abl['full'] - abl['panel_only'])*100:+.1f} pp",
            })
        fam_names = sorted({f for a in attr_by_target.values() for f in a})
        fam_colors = {"Demographics": "#64748b", "Cardiovascular": "#ef4444",
                      "Diabetes Markers": "#f59e0b", "Liver Function": "#84cc16",
                      "Kidney Function": "#06b6d4", "Other": "#a78bfa"}
        fig_cd = go.Figure()
        for fam in fam_names:
            fig_cd.add_trace(go.Bar(
                name=fam,
                x=[DISEASE_NAMES[TARGET_COLS.index(t)] for t in TARGET_COLS],
                y=[attr_by_target[t].get(fam, 0.0) for t in TARGET_COLS],
                marker_color=fam_colors.get(fam, "#a78bfa"),
            ))
        fig_cd.update_layout(
            barmode="group", height=320, margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="SHAP contribution by feature family (log-odds)", font=dict(size=13, color="#0f172a")),
            yaxis_title="contribution", legend=dict(orientation="h", y=-0.25),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_cd, use_container_width=True)
        st.markdown("**Ablation — same patient scored three ways**")
        st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)
        st.caption(
            "“Own markers only” keeps demographics + that disease's lab markers; "
            "“removed” strips them. A large gap = the score is driven by its own markers; "
            "a small gap = cross-disease features carry most of the signal."
        )
    except Exception as e:
        st.warning(f"Cross-disease analysis unavailable: {e}")

if missing_used:
    st.markdown("""<div class="missing-box"><strong>ℹ️ Missing Features</strong><br>
        Not provided, auto-handled via sentinel encoding (-999):<br><br>"""
        + " · ".join(f"<code>{FEATURE_LABELS.get(f,f)}</code>" for f in missing_used)
        + "</div>", unsafe_allow_html=True)

st.markdown("")
st.caption("⚠️ **Disclaimer:** For research purposes only. Not a substitute for professional medical diagnosis or advice. Always consult a qualified healthcare provider.")
