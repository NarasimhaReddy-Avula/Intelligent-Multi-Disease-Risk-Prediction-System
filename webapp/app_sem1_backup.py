"""
Multi-Disease Risk Prediction — Streamlit Frontend
Place this file at: webapp/app.py
Run with: streamlit run webapp/app.py
"""

import sys
import os
import json
import hashlib
from pathlib import Path

# ── Project root resolution ───────────────────────────────────────────────────
# webapp/app.py is one level inside the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
import streamlit as st
import streamlit.components.v1 as components

from src.explainability.shap_payload import build_structured_shap_factors
from src.llm.clients import load_openai_compatible_config_from_env
from src.llm.explain import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DISCLAIMER as LLM_DISCLAIMER,
    build_patient_payload_for_llm,
    generate_ai_chat_reply,
    generate_ai_explanation,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACT_SUBDIR = Path('week7_artifacts')
SCALER_FILENAME = 'scaler.pkl'
SUMMARY_FILENAME = 'summary.json'

XGB_MODEL_FILES = {
    'diabetes_risk': 'diabetes_XGBoost.pkl',
    'heart_disease_risk': 'heart_XGBoost.pkl',
    'liver_disease_risk': 'liver_XGBoost.pkl',
    'kidney_disease_risk': 'kidney_XGBoost.pkl',
}

SENTINEL_MISSING_VALUE = -999.0

# Global explainability uses a small sample from an existing dataset file.
# This keeps runtime/memory reasonable while giving stable global feature rankings.
GLOBAL_EXPLAIN_MAX_ROWS = 2000

# Ordered by preference (scaled + sentinel first).
REFERENCE_DATASET_CANDIDATES = [
    PROJECT_ROOT / 'datasets' / 'combined_unified_sentinel_scaled.csv',
    PROJECT_ROOT / 'datasets' / 'combined_unified_masked_scaled.csv',
    PROJECT_ROOT / 'datasets' / 'combined_unified_sentinel.csv',
    PROJECT_ROOT / 'datasets' / 'combined_unified_masked.csv',
]

TARGET_COLS   = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
DISEASE_NAMES = ['Diabetes', 'Heart Disease', 'Liver Disease', 'Kidney Disease']
DISEASE_ICONS = ['🩸', '❤️', '🫁', '🫘']

NUMERIC_COLS = [
    'age', 'gender', 'bmi',
    'systolic_bp', 'diastolic_bp',
    'cholesterol_total', 'glucose_fasting', 'hba1c',
    'hdl', 'ldl', 'triglycerides',
    'alt', 'ast', 'alp', 'albumin', 'total_protein',
    'bilirubin_total', 'bilirubin_direct', 'ag_ratio',
    'serum_creatinine', 'bun', 'gfr', 'protein_urine', 'hemoglobin',
]

# Human-readable labels for each feature
FEATURE_LABELS = {
    'age':              'Age (years)',
    'gender':           'Gender',
    'bmi':              'BMI (kg/m²)',
    'systolic_bp':      'Systolic BP (mmHg)',
    'diastolic_bp':     'Diastolic BP (mmHg)',
    'cholesterol_total':'Total Cholesterol (mg/dL)',
    'glucose_fasting':  'Fasting Glucose (mg/dL)',
    'hba1c':            'HbA1c (%)',
    'hdl':              'HDL Cholesterol (mg/dL)',
    'ldl':              'LDL Cholesterol (mg/dL)',
    'triglycerides':    'Triglycerides (mg/dL)',
    'alt':              'ALT / SGPT (U/L)',
    'ast':              'AST / SGOT (U/L)',
    'alp':              'ALP — Alkaline Phosphatase (U/L)',
    'albumin':          'Albumin (g/dL)',
    'total_protein':    'Total Protein (g/dL)',
    'bilirubin_total':  'Total Bilirubin (mg/dL)',
    'bilirubin_direct': 'Direct Bilirubin (mg/dL)',
    'ag_ratio':         'A/G Ratio',
    'serum_creatinine': 'Serum Creatinine (mg/dL)',
    'bun':              'BUN — Blood Urea Nitrogen (mg/dL)',
    'gfr':              'GFR — Glomerular Filtration Rate (mL/min)',
    'protein_urine':    'Urine Protein (g/dL)',
    'hemoglobin':       'Hemoglobin (g/dL)',
}

# Grouped by clinical domain for organized UI
FEATURE_GROUPS = {
    '👤 Demographics': ['age', 'gender', 'bmi'],
    '🫀 Cardiovascular': ['systolic_bp', 'diastolic_bp', 'cholesterol_total', 'hdl', 'ldl', 'triglycerides'],
    '🩸 Diabetes Markers': ['glucose_fasting', 'hba1c'],
    '🫁 Liver Function': ['alt', 'ast', 'alp', 'albumin', 'total_protein', 'bilirubin_total', 'bilirubin_direct', 'ag_ratio'],
    '🫘 Kidney Function': ['serum_creatinine', 'bun', 'gfr', 'protein_urine', 'hemoglobin'],
}

# Normal ranges for reference display
NORMAL_RANGES = {
    'age':              (0, 120, 'years'),
    'bmi':              (18.5, 24.9, 'kg/m²'),
    'systolic_bp':      (90, 120, 'mmHg'),
    'diastolic_bp':     (60, 80, 'mmHg'),
    'cholesterol_total':(0, 200, 'mg/dL'),
    'glucose_fasting':  (70, 100, 'mg/dL'),
    'hba1c':            (4.0, 5.6, '%'),
    'hdl':              (40, 60, 'mg/dL'),
    'ldl':              (0, 100, 'mg/dL'),
    'triglycerides':    (0, 150, 'mg/dL'),
    'alt':              (7, 40, 'U/L'),
    'ast':              (10, 40, 'U/L'),
    'alp':              (44, 147, 'U/L'),
    'albumin':          (3.5, 5.0, 'g/dL'),
    'total_protein':    (6.0, 8.3, 'g/dL'),
    'bilirubin_total':  (0.1, 1.2, 'mg/dL'),
    'bilirubin_direct': (0.0, 0.3, 'mg/dL'),
    'ag_ratio':         (1.0, 2.5, ''),
    'serum_creatinine': (0.6, 1.2, 'mg/dL'),
    'bun':              (7, 20, 'mg/dL'),
    'gfr':              (90, 120, 'mL/min'),
    'protein_urine':    (0.0, 0.15, 'g/dL'),
    'hemoglobin':       (12.0, 17.5, 'g/dL'),
}

RISK_COLORS = {
    'LOW':    ('#22c55e', '#dcfce7'),
    'MEDIUM': ('#f59e0b', '#fef3c7'),
    'HIGH':   ('#ef4444', '#fee2e2'),
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_artifact_paths():
    """Find XGBoost artifacts even when the app is launched from nested copies."""
    search_roots = [PROJECT_ROOT]
    # Covers layouts like <workspace>/tmp/gh_repo/webapp/app.py
    search_roots.extend(PROJECT_ROOT.parents[:4])

    tried_dirs = []
    for root in search_roots:
        artifact_dir = root / ARTIFACT_SUBDIR
        scaler_path = artifact_dir / SCALER_FILENAME
        summary_path = artifact_dir / SUMMARY_FILENAME
        model_paths = {t: artifact_dir / fname for t, fname in XGB_MODEL_FILES.items()}
        tried_dirs.append(str(artifact_dir))
        if scaler_path.exists() and all(p.exists() for p in model_paths.values()):
            return artifact_dir, scaler_path, model_paths, (summary_path if summary_path.exists() else None)

    searched = "\n".join(f"- {p}" for p in tried_dirs)
    raise FileNotFoundError(
        "Could not find required XGBoost artifacts (scaler.pkl + 4 *_XGBoost.pkl files). "
        f"Searched:\n{searched}"
    )


def _safe_mean_auc_from_summary(summary_path):
    if summary_path is None:
        return None
    try:
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        xgb_aucs = data.get('xgb_aucs')
        if isinstance(xgb_aucs, dict) and xgb_aucs:
            vals = [float(v) for v in xgb_aucs.values()]
            return float(np.mean(vals))
    except Exception:
        return None
    return None


@st.cache_resource
def load_xgb_models_and_scaler():
    """Load XGBoost models + scaler once, cache for session."""
    artifact_dir, scaler_path, model_paths, summary_path = resolve_artifact_paths()

    scaler_bundle = joblib.load(scaler_path)
    if not isinstance(scaler_bundle, dict):
        raise TypeError('scaler.pkl format is invalid: expected dict bundle.')

    if 'scaler' not in scaler_bundle:
        raise KeyError("scaler.pkl is missing key 'scaler'.")
    if 'numeric_cols' not in scaler_bundle:
        raise KeyError("scaler.pkl is missing key 'numeric_cols'.")

    numeric_cols = list(scaler_bundle['numeric_cols'])
    if not numeric_cols:
        raise ValueError('numeric_cols is empty in scaler bundle.')

    xgb_models = {t: joblib.load(p) for t, p in model_paths.items()}
    mean_auc = _safe_mean_auc_from_summary(summary_path)
    meta = {
        'artifact_dir': str(artifact_dir),
        'mean_auc': mean_auc,
        'summary_path': (str(summary_path) if summary_path is not None else None),
    }

    return xgb_models, scaler_bundle, numeric_cols, meta


def patient_to_xgb(patient_dict, numeric_cols, scaler, scaler_cols):
    """Convert raw clinical values → sentinel-encoded numpy array for XGBoost."""
    raw = np.array([[
        float(patient_dict[c]) if (c in patient_dict and patient_dict[c] is not None)
        else scaler.mean_[i]
        for i, c in enumerate(scaler_cols)
    ]], dtype=np.float32)
    scaled = scaler.transform(raw)[0]
    row = {
        col: float(scaled[i]) if (patient_dict.get(col) is not None) else SENTINEL_MISSING_VALUE
        for i, col in enumerate(scaler_cols)
    }
    return np.array([[row.get(c, SENTINEL_MISSING_VALUE) for c in numeric_cols]], dtype=np.float32)


def _predict_proba_binary(model, X):
    """Return P(y=1) for a binary classifier-like model."""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        if proba.ndim == 1:
            return float(proba[0])
    # Fallback for raw Booster-like objects.
    try:
        import xgboost as xgb

        dmat = xgb.DMatrix(X)
        pred = model.predict(dmat)
        pred = np.asarray(pred)
        return float(pred[0])
    except Exception as e:
        raise TypeError('Loaded XGBoost artifact does not support probability prediction.') from e


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _xgb_pred_contribs(model, X: np.ndarray, feature_names):
    """Return TreeSHAP-like per-feature contributions using XGBoost pred_contribs.

    Notes:
    - Contributions are in *margin/log-odds* space for binary:logistic models.
    - The last column is the bias term.
    """
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, 'get_booster') else model
    dmat = xgb.DMatrix(X, feature_names=list(feature_names))

    contribs = booster.predict(dmat, pred_contribs=True)
    contribs = np.asarray(contribs)
    if contribs.ndim != 2 or contribs.shape[0] != X.shape[0] or contribs.shape[1] != (X.shape[1] + 1):
        raise ValueError(
            'Unexpected pred_contribs shape. '
            f'Got {tuple(contribs.shape)}; expected ({X.shape[0]}, {X.shape[1] + 1}).'
        )

    shap_vals = contribs[:, :-1]
    bias = contribs[:, -1]
    return shap_vals, bias


def _try_import_shap():
    """Lazily import SHAP + Matplotlib.

    Returns:
        (shap_module, matplotlib_pyplot, error)
    """
    try:
        import shap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        return shap, plt, None
    except Exception as e:
        return None, None, e


def _display_feature_names(feature_names):
    return [FEATURE_LABELS.get(f, f) for f in feature_names]


def _patient_signature(patient_dict, feature_names) -> str:
    """Stable signature for caching per-patient explainability across reruns."""
    payload = {f: patient_dict.get(f) for f in feature_names}
    dumped = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(dumped.encode('utf-8')).hexdigest()


def _patient_raw_vector(patient_dict, feature_names) -> np.ndarray:
    """Raw values in model feature order (NaN for missing) for display in SHAP plots."""
    vals = []
    for f in feature_names:
        v = patient_dict.get(f)
        if v is None:
            vals.append(np.nan)
        else:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
    return np.asarray(vals, dtype=np.float32)


def _inverse_standardize_with_sentinel(X_scaled: np.ndarray, scaler, sentinel: float) -> np.ndarray:
    """Inverse StandardScaler transform, keeping sentinel-missing values as NaN for readability."""
    X = np.asarray(X_scaled, dtype=np.float32)
    mask = (X == float(sentinel))
    X2 = X.copy()
    X2[mask] = 0.0

    mean = getattr(scaler, 'mean_', None)
    scale = getattr(scaler, 'scale_', None)
    if mean is not None and scale is not None:
        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        if mean.shape[0] == X.shape[1] and scale.shape[0] == X.shape[1]:
            X_raw = (X2 * scale) + mean
            X_raw = np.asarray(X_raw, dtype=np.float32)
            X_raw[mask] = np.nan
            return X_raw

    try:
        X_raw = np.asarray(scaler.inverse_transform(X2), dtype=np.float32)
        X_raw[mask] = np.nan
        return X_raw
    except Exception:
        # Fall back to showing scaled values if inverse transform isn't available.
        X_disp = X.astype(np.float32)
        X_disp[mask] = np.nan
        return X_disp


def _impact_adjective(share_pct: float) -> str:
    # Heuristic buckets for non-technical text.
    if share_pct >= 25.0:
        return 'significantly'
    if share_pct >= 12.0:
        return 'moderately'
    return 'slightly'


def _format_raw_value(feat: str, raw_val) -> str:
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return 'not provided'
    if feat == 'gender':
        return 'Male' if float(raw_val) >= 0.5 else 'Female'
    nr = NORMAL_RANGES.get(feat)
    unit = ''
    if nr:
        _lo, _hi, unit = nr
    try:
        v = float(raw_val)
        if unit:
            return f"{v:.2f} {unit}" if abs(v) < 100 else f"{v:.1f} {unit}"
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"
    except Exception:
        return str(raw_val)


def _feature_level_phrase(feat: str, raw_val) -> str:
    label = FEATURE_LABELS.get(feat, feat)
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return f"Missing {label}"
    if feat == 'gender':
        return f"Gender ({_format_raw_value(feat, raw_val)})"
    nr = NORMAL_RANGES.get(feat)
    if nr is None:
        return label
    lo, hi, _unit = nr
    try:
        v = float(raw_val)
    except Exception:
        return label
    if v > hi:
        return f"High {label}"
    if v < lo:
        return f"Low {label}"
    return label


def _plain_english_summary(disease_name: str, feature_names, shap_row: np.ndarray, patient_dict, top_n: int = 3):
    """Return (increasing, decreasing) human-readable factor summaries."""
    shap_row = np.asarray(shap_row, dtype=np.float32)
    abs_vals = np.abs(shap_row)
    total = float(abs_vals.sum()) or 1.0
    shares = (abs_vals / total) * 100.0
    order = np.argsort(abs_vals)[::-1]

    inc = []
    dec = []
    for idx in order:
        feat = feature_names[int(idx)]
        contrib = float(shap_row[int(idx)])
        if contrib == 0.0:
            continue
        share = float(shares[int(idx)])
        adjective = _impact_adjective(share)
        raw_val = patient_dict.get(feat)
        phrase = _feature_level_phrase(feat, raw_val)
        val_str = _format_raw_value(feat, raw_val)

        if contrib > 0 and len(inc) < top_n:
            inc.append(f"{phrase} ({val_str}) increased {disease_name} risk {adjective}.")
        elif contrib < 0 and len(dec) < top_n:
            dec.append(f"{phrase} ({val_str}) decreased {disease_name} risk {adjective}.")
        if len(inc) >= top_n and len(dec) >= top_n:
            break

    return inc, dec


@st.cache_data
def _find_reference_dataset_path(numeric_cols_tuple: tuple[str, ...]):
    """Pick the first dataset file that contains all required model feature columns."""
    import pandas as pd

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
def _compute_global_shap_bundle(
    model_sigs: dict,
    dataset_sig: tuple,
    numeric_cols_tuple: tuple[str, ...],
    max_rows: int,
):
    """Compute global SHAP (TreeSHAP via XGBoost pred_contribs) for all diseases.

    Cached by:
    - model_sigs (paths + timestamps)
    - dataset_sig (path + timestamp)
    - numeric_cols_tuple + max_rows
    """
    import pandas as pd
    import xgboost as xgb

    dataset_path = dataset_sig[0]
    df = pd.read_csv(dataset_path, usecols=list(numeric_cols_tuple), nrows=int(max_rows))
    X_ref = df.to_numpy(dtype=np.float32)

    dmat = xgb.DMatrix(X_ref, feature_names=list(numeric_cols_tuple))

    bundle = {}
    for target, (model_path, _mtime, _size) in model_sigs.items():
        model = joblib.load(model_path)
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        contribs = booster.predict(dmat, pred_contribs=True)
        contribs = np.asarray(contribs)
        if contribs.ndim != 2 or contribs.shape[1] != (X_ref.shape[1] + 1):
            raise ValueError(
                f"Unexpected pred_contribs shape for {target}: {tuple(contribs.shape)}; "
                f"expected (n, {X_ref.shape[1] + 1})."
            )
        bundle[target] = {
            'shap_vals': contribs[:, :-1].astype(np.float32),
            'bias': contribs[:, -1].astype(np.float32),
        }

    return X_ref, bundle


def predict(patient_dict, xgb_models, scaler_bundle, numeric_cols):
    """Run XGBoost inference and return dict of disease → probability."""
    scaler = scaler_bundle['scaler']
    scaler_cols = list(scaler_bundle['numeric_cols'])

    X = patient_to_xgb(patient_dict, numeric_cols, scaler, scaler_cols)
    return {
        target: _predict_proba_binary(xgb_models[target], X)
        for target in TARGET_COLS
    }


def risk_level(prob):
    if prob >= 0.65:   return 'HIGH'
    elif prob >= 0.35: return 'MEDIUM'
    else:              return 'LOW'


def _mentioned_disease_targets(text: str) -> list[str]:
    """Heuristic mapping from user question to a disease target.

    Used by the AI chat so questions like "what about kidney?" use the kidney model context.
    """
    t = (text or "").lower()
    hits: list[str] = []

    def has_any(words: list[str]) -> bool:
        return any(w in t for w in words)

    # 1) Strong signal: explicit disease mention
    explicit = {
        'kidney_disease_risk': ["kidney", "renal", "nephro"],
        'liver_disease_risk': ["liver", "hepatic", "hepat"],
        'heart_disease_risk': ["heart", "cardio", "cardiac", "cardiovascular", "coronary"],
        'diabetes_risk': ["diabetes", "diabetic"],
    }
    explicit_hits = [target for target, words in explicit.items() if has_any(words)]
    if len(explicit_hits) == 1:
        return explicit_hits
    if len(explicit_hits) > 1:
        hits.extend(explicit_hits)

    # 2) Secondary signal: disease-specific lab markers
    if has_any(["gfr", "creatinine", "bun", "proteinuria", "albuminuria", "urine protein"]):
        hits.append('kidney_disease_risk')
    if has_any(["alt", "ast", "alp", "bilirubin", "jaundice"]):
        hits.append('liver_disease_risk')
    if has_any(["hba1c", "a1c", "glucose", "fasting glucose"]):
        hits.append('diabetes_risk')
    if has_any(["ldl", "hdl", "triglycer", "cholesterol"]):
        hits.append('heart_disease_risk')

    # Preserve original order, de-dup
    uniq: list[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    return uniq


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='MedRisk AI — Multi-Disease Risk Predictor',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c75 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    color: #f0f9ff;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #7dd3fc;
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.2);
    border: 1px solid rgba(56,189,248,0.4);
    color: #7dd3fc;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    margin-top: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Risk cards */
.risk-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.risk-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.3rem;
}
.risk-card-disease {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    font-weight: 400;
    margin-bottom: 0.5rem;
}
.risk-card-prob {
    font-size: 2.8rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.risk-card-level {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Progress bar custom */
.prob-bar-container {
    background: rgba(0,0,0,0.08);
    border-radius: 4px;
    height: 6px;
    margin: 0.5rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* Feature input section */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
    margin-top: 1.5rem;
}

/* Missing info box */
.missing-box {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #64748b;
}

/* Warning banner */
.warning-banner {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #92400e;
    margin-top: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}
/* Keep typed/selected values readable on white input controls in sidebar */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input {
    color: #000000 !important;
}

/* Submit button */
div.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    width: 100%;
    transition: all 0.2s;
    cursor: pointer;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(3,105,161,0.4);
}

/* Normal range tag */
.normal-tag {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: -0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
try:
    xgb_models, scaler_bundle, numeric_cols, meta = load_xgb_models_and_scaler()
    model_loaded = True
    model_auc    = meta.get('mean_auc')
    artifact_dir = meta.get('artifact_dir')
except Exception as e:
    model_loaded = False
    model_error  = str(e)
    xgb_models = {}
    scaler_bundle = None
    numeric_cols = NUMERIC_COLS


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MedRisk AI</h1>
    <p>Intelligent Multi-Disease Risk Prediction Platform</p>
    <span class="badge">XGBoost · Week 7 Artifacts · IIIT Sri City BTP 2025–26</span>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model could not be loaded: {model_error}")
    st.info(
        "Expected artifacts: week7_artifacts/scaler.pkl and "
        "week7_artifacts/{diabetes,heart,liver,kidney}_XGBoost.pkl"
    )
    st.stop()

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Model", "XGBoost")
with col_info2:
    st.metric("Mean AUC", f"{model_auc:.4f}" if model_auc is not None else "n/a")
with col_info3:
    st.metric("Artifacts", "week7_artifacts")

if model_loaded and set(numeric_cols) != set(NUMERIC_COLS):
    st.warning(
        "Artifact feature list differs from UI feature list. "
        "Predictions will use the artifact order; consider syncing NUMERIC_COLS if needed."
    )

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PATIENT INFO
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Patient Information")
    patient_name = st.text_input("Patient Name", value="", placeholder="e.g. Rahul Sharma")
    patient_age_sidebar = st.number_input("Age (quick entry)", min_value=1, max_value=120, value=45)
    patient_gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    st.markdown("""
    **How it works:**
    - Select the lab tests available
    - Enter the values
    - Missing tests are handled automatically
    - Model predicts risk for all 4 diseases

    **Diseases predicted:**
    - 🩸 Diabetes
    - ❤️ Heart Disease
    - 🫁 Liver Disease
    - 🫘 Kidney Disease

    **Risk levels:**
    - 🟢 LOW — below 35%
    - 🟡 MEDIUM — 35% to 65%
    - 🔴 HIGH — above 65%
    """)
    st.markdown("---")
    st.caption("⚠️ For clinical research only. Not a substitute for medical advice.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🧪 Enter Lab Values")
st.markdown("Select which tests are available and enter the values. Leave unselected fields blank — the model handles missing data automatically.")

# Track which features user wants to enter
patient_dict = {}

# Gender from sidebar
patient_dict['gender'] = 1.0 if patient_gender == "Male" else 0.0
patient_dict['age']    = float(patient_age_sidebar)

for group_name, features in FEATURE_GROUPS.items():
    if group_name == '👤 Demographics':
        # Already handled via sidebar
        with st.expander(f"{group_name} — already entered above", expanded=False):
            st.info(f"Age: {patient_age_sidebar} | Gender: {patient_gender}")
            # BMI
            enter_bmi = st.checkbox("Enter BMI", value=True, key="cb_bmi")
            if enter_bmi:
                bmi_val = st.number_input(
                    FEATURE_LABELS['bmi'], min_value=10.0, max_value=80.0,
                    value=25.0, step=0.1, key="bmi"
                )
                st.markdown(f'<div class="normal-tag">Normal: 18.5 – 24.9 kg/m²</div>',
                            unsafe_allow_html=True)
                patient_dict['bmi'] = bmi_val
            else:
                patient_dict['bmi'] = None
        continue

    with st.expander(f"{group_name}", expanded=True):
        cols = st.columns(2)
        for idx, feat in enumerate(features):
            with cols[idx % 2]:
                enter = st.checkbox(
                    f"Enter {FEATURE_LABELS[feat]}",
                    value=False,
                    key=f"cb_{feat}"
                )
                if enter:
                    nr = NORMAL_RANGES.get(feat)
                    if feat == 'gender':
                        continue
                    if nr:
                        lo, hi, _ = nr
                        min_v = 0.0
                        default_v = float((lo + hi) / 2)
                        # Build a safe UI range from the normal upper bound.
                        # This avoids max=0 for features whose normal lower bound is 0.
                        max_v = max(float(hi) * 5.0, default_v + 1.0, 10.0)
                    else:
                        min_v = 0.0
                        default_v = 0.0
                        max_v = 1000.0
                    val = st.number_input(
                        FEATURE_LABELS[feat],
                        min_value=min_v,
                        max_value=float(max_v),
                        value=default_v,
                        step=0.1,
                        key=f"val_{feat}",
                        label_visibility="collapsed"
                    )
                    if nr:
                        lo, hi, unit = nr
                        flag = "✅" if lo <= val <= hi else "⚠️ Outside normal"
                        st.markdown(
                            f'<div class="normal-tag">Normal: {lo}–{hi} {unit} {flag}</div>',
                            unsafe_allow_html=True
                        )
                    patient_dict[feat] = val
                else:
                    patient_dict[feat] = None

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY OF ENTERED VALUES
# ─────────────────────────────────────────────────────────────────────────────
provided = [f for f in numeric_cols if patient_dict.get(f) is not None]
missing  = [f for f in numeric_cols if patient_dict.get(f) is None]

st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**✅ Provided:** {len(provided)} / {len(NUMERIC_COLS)} features")
    if provided:
        st.caption(", ".join(FEATURE_LABELS.get(f, f) for f in provided))
with col_b:
    st.markdown(f"**⚪ Missing (auto-handled):** {len(missing)} features")
    if missing:
        st.caption(", ".join(FEATURE_LABELS.get(f, f) for f in missing))

if len(provided) < 3:
    st.markdown("""
    <div class="warning-banner">
    ⚠️ Very few features provided. Predictions will rely heavily on demographic data.
    Consider entering at least a few lab values for meaningful results.
    </div>
    """, unsafe_allow_html=True)

st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────

# Persist the latest prediction so explainability widgets (slider/tabs) work
# across Streamlit reruns without requiring repeated clicks.
if 'pred_results' not in st.session_state:
    st.session_state['pred_results'] = None
    st.session_state['pred_patient'] = None
    st.session_state['pred_name'] = None

predict_clicked = st.button("🔍 Predict Disease Risk", type="primary")

if predict_clicked:
    with st.spinner("Running XGBoost inference..."):
        try:
            results = predict(patient_dict, xgb_models, scaler_bundle, numeric_cols)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    st.session_state['pred_results'] = results
    st.session_state['pred_patient'] = dict(patient_dict)
    st.session_state['pred_name'] = patient_name.strip() if patient_name.strip() else "Patient"


if st.session_state.get('pred_results') is not None:
    results = st.session_state['pred_results']
    patient_used = st.session_state.get('pred_patient') or dict(patient_dict)
    pname = st.session_state.get('pred_name') or "Patient"

    provided_used = [f for f in numeric_cols if patient_used.get(f) is not None]
    missing_used = [f for f in numeric_cols if patient_used.get(f) is None]

    st.markdown("---")
    st.markdown(f"## 📊 Risk Assessment — {pname}")
    st.markdown(f"*{len(provided_used)} features provided · {len(missing_used)} auto-handled*")
    st.caption("If you changed inputs above, click Predict again to refresh.")
    st.markdown("")

    # ── Risk cards ────────────────────────────────────────────────────────────
    cols = st.columns(4)
    for i, (target, name, icon) in enumerate(zip(TARGET_COLS, DISEASE_NAMES, DISEASE_ICONS)):
        prob  = results[target]
        level = risk_level(prob)
        color, bg = RISK_COLORS[level]

        with cols[i]:
            st.markdown(f"""
            <div class="risk-card" style="
                background: {bg};
                border-color: {color}40;
                color: #0f172a;
            ">
                <div class="risk-card-title">{icon} Disease Risk</div>
                <div class="risk-card-disease">{name}</div>
                <div class="risk-card-prob" style="color: {color};">{prob*100:.1f}%</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:{color};"></div>
                </div>
                <div class="risk-card-level" style="color:{color};">{level} RISK</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Explainability (local) ───────────────────────────────────────────────
    with st.expander("🔎 Explainability — Top Feature Drivers", expanded=False):
        st.markdown(
            "Shows which inputs pushed each disease risk **up** or **down** for this patient. "
            "Contributions are computed with XGBoost `pred_contribs` (TreeSHAP). "
            "The numeric contributions are additive in **log-odds** space (positive increases risk; negative decreases it). "
            "If the `shap` package is available in your current environment, the same values are also shown as SHAP plots."
        )

        max_k = min(20, len(numeric_cols))
        if max_k >= 3:
            top_k = st.slider(
                "Top features to show",
                min_value=3,
                max_value=max_k,
                value=min(10, max_k),
                step=1,
                key="exp_top_k",
            )
        else:
            top_k = max_k

        scaler = scaler_bundle['scaler']
        scaler_cols = list(scaler_bundle['numeric_cols'])
        X = patient_to_xgb(patient_used, numeric_cols, scaler, scaler_cols)

        # Cache local SHAP per patient snapshot to keep UI snappy across reruns.
        local_cache_key = f"{_patient_signature(patient_used, numeric_cols)}::{meta.get('artifact_dir', '')}"
        if st.session_state.get('local_shap_cache_key') != local_cache_key:
            st.session_state['local_shap_cache_key'] = local_cache_key
            st.session_state['local_shap_cache'] = {}
        local_cache = st.session_state.get('local_shap_cache', {})

        raw_vec = _patient_raw_vector(patient_used, numeric_cols)
        display_names = _display_feature_names(numeric_cols)

        tabs = st.tabs([f"{icon} {name}" for icon, name in zip(DISEASE_ICONS, DISEASE_NAMES)])
        for tab, target, disease_name in zip(tabs, TARGET_COLS, DISEASE_NAMES):
            with tab:
                try:
                    if target in local_cache:
                        shap_row = np.asarray(local_cache[target]['shap_row'], dtype=np.float32)
                        bias_val = float(local_cache[target]['bias_val'])
                    else:
                        shap_vals, bias = _xgb_pred_contribs(xgb_models[target], X, numeric_cols)
                        shap_row = np.asarray(shap_vals[0], dtype=np.float32)
                        bias_val = float(bias[0])
                        local_cache[target] = {
                            'shap_row': shap_row,
                            'bias_val': bias_val,
                        }
                        st.session_state['local_shap_cache'] = local_cache

                    import pandas as pd

                    rows = []
                    for feat, contrib in zip(numeric_cols, shap_row):
                        raw_val = patient_used.get(feat)
                        rows.append({
                            'Feature': FEATURE_LABELS.get(feat, feat),
                            'Raw value': (float(raw_val) if raw_val is not None else 'missing'),
                            'Contribution (log-odds)': float(contrib),
                        })

                    df_exp = pd.DataFrame(rows)
                    df_exp['Abs contribution'] = df_exp['Contribution (log-odds)'].abs()
                    denom = float(df_exp['Abs contribution'].sum()) or 1.0
                    df_exp['Impact share (%)'] = (df_exp['Abs contribution'] / denom) * 100.0
                    df_exp = df_exp.sort_values('Abs contribution', ascending=False)
                    df_top = df_exp.head(top_k)[
                        ['Feature', 'Raw value', 'Contribution (log-odds)', 'Impact share (%)']
                    ]
                    df_top['Impact share (%)'] = df_top['Impact share (%)'].round(1)

                    st.dataframe(df_top, use_container_width=True, hide_index=True)

                    margin = bias_val + float(np.sum(shap_row))
                    explained_prob = _sigmoid(margin)
                    base_prob = _sigmoid(bias_val)

                    st.caption(
                        f"Base risk≈{base_prob * 100:.1f}% · "
                        f"Explained risk≈{explained_prob * 100:.1f}% · "
                        f"Model risk={results[target] * 100:.1f}%"
                    )

                    certainty_proxy = float(abs(float(results[target]) - 0.5) * 2.0)
                    st.caption(
                        f"Certainty proxy≈{certainty_proxy * 100:.0f}% (heuristic: distance from 50%)."
                    )

                    inc, dec = _plain_english_summary(
                        disease_name=disease_name,
                        feature_names=list(numeric_cols),
                        shap_row=shap_row,
                        patient_dict=patient_used,
                        top_n=min(3, int(top_k)),
                    )
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown("**Plain English — Factors increasing risk**")
                        if inc:
                            st.markdown("\n".join(f"- {s}" for s in inc))
                        else:
                            st.caption("No strong risk-increasing factors detected.")
                    with col_t2:
                        st.markdown("**Plain English — Factors decreasing risk**")
                        if dec:
                            st.markdown("\n".join(f"- {s}" for s in dec))
                        else:
                            st.caption("No strong risk-decreasing factors detected.")

                    shap, plt, shap_err = _try_import_shap()
                    if shap is None:
                        st.info(
                            "SHAP plots are unavailable because `shap` (or `matplotlib`) could not be imported in the "
                            "current Python environment. The table above is still the same TreeSHAP explanation."
                        )
                        with st.expander("Troubleshooting (environment)", expanded=False):
                            st.code(
                                "Python executable:\n"
                                f"{sys.executable}\n\n"
                                "VIRTUAL_ENV:\n"
                                f"{os.environ.get('VIRTUAL_ENV', '(not set)')}\n\n"
                                "Import error:\n"
                                f"{shap_err}"
                            )
                            st.code(
                                "If you have a venv with SHAP installed, restart Streamlit using that Python, e.g.:\n"
                                "  run_with_env.bat -m streamlit run webapp/app.py\n"
                                "or:\n"
                                "  C:\\dev\\venvs\\btp_env\\Scripts\\python.exe -m streamlit run webapp/app.py"
                            )
                    else:
                        try:
                            exp = shap.Explanation(
                                values=np.asarray(shap_row, dtype=np.float32),
                                base_values=float(bias_val),
                                data=np.asarray(raw_vec, dtype=np.float32),
                                feature_names=display_names,
                            )

                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                st.markdown("**SHAP bar — feature impact magnitude**")
                                plt.figure()
                                shap.plots.bar(exp, max_display=int(top_k), show=False)
                                fig = plt.gcf()
                                st.pyplot(fig, use_container_width=True, clear_figure=True)

                            with col_p2:
                                st.markdown("**SHAP waterfall — how the prediction adds up**")
                                plt.figure()
                                shap.plots.waterfall(exp, max_display=int(top_k), show=False)
                                fig = plt.gcf()
                                st.pyplot(fig, use_container_width=True, clear_figure=True)

                            st.markdown("**SHAP force — interactive (per patient)**")
                            force = shap.force_plot(
                                float(bias_val),
                                np.asarray(shap_row, dtype=np.float32),
                                np.asarray(raw_vec, dtype=np.float32),
                                feature_names=display_names,
                                matplotlib=False,
                            )
                            components.html(
                                f"<head>{shap.getjs()}</head><body>{force.html()}</body>",
                                height=280,
                                scrolling=True,
                            )

                            st.caption(
                                "Note: SHAP values are computed on the model's standardized inputs, but the plots above "
                                "display raw clinical values for readability. Missing values are shown as blank/NaN."
                            )
                        except Exception as e:
                            st.warning(f"Could not render SHAP plots for {disease_name}: {e}")
                except Exception as e:
                    st.warning(f"Explainability unavailable for {disease_name}: {e}")

    # ── Explainability (global) ──────────────────────────────────────────────
    with st.expander("🌍 Explainability — Global Feature Importance", expanded=False):
        st.markdown(
            "Shows which features are most influential **overall** for each disease model, estimated by "
            "mean(|SHAP|) on a small sample from an existing dataset file."
        )

        max_g = min(30, len(numeric_cols))
        top_k_global = st.slider(
            "Top features to show (global)",
            min_value=min(5, max_g) if max_g else 1,
            max_value=max_g if max_g else 1,
            value=min(15, max_g) if max_g else 1,
            step=1,
            key="exp_top_k_global",
        )

        shap, plt, shap_err = _try_import_shap()
        if shap is None:
            st.info(
                "Global SHAP plots require `shap` + `matplotlib` in the current Python environment. "
                "If you don't want to install anything, you can still use the per-patient table above."
            )
            with st.expander("Troubleshooting (environment)", expanded=False):
                st.code(
                    "Python executable:\n"
                    f"{sys.executable}\n\n"
                    "VIRTUAL_ENV:\n"
                    f"{os.environ.get('VIRTUAL_ENV', '(not set)')}\n\n"
                    "Import error:\n"
                    f"{shap_err}"
                )
        else:
            ref_path = _find_reference_dataset_path(tuple(numeric_cols))
            if ref_path is None:
                st.warning(
                    "Could not find a reference dataset file that contains all required feature columns. "
                    "Expected one of: combined_unified_sentinel_scaled.csv / combined_unified_masked_scaled.csv (under datasets/)."
                )
            else:
                ref_p = Path(ref_path)
                try:
                    ds_stat = ref_p.stat()
                except Exception as e:
                    st.warning(f"Reference dataset could not be accessed: {e}")
                else:
                    dataset_sig = (str(ref_p), float(ds_stat.st_mtime), int(ds_stat.st_size))

                    artifact_dir_path = Path(meta.get('artifact_dir', ''))
                    model_sigs = {}
                    missing_models = []
                    for target, fname in XGB_MODEL_FILES.items():
                        mp = artifact_dir_path / fname
                        if not mp.exists():
                            missing_models.append(fname)
                            continue
                        st_mp = mp.stat()
                        model_sigs[target] = (str(mp), float(st_mp.st_mtime), int(st_mp.st_size))

                    if missing_models:
                        st.warning("Missing model files for global explainability: " + ", ".join(missing_models))
                    if model_sigs:
                        try:
                            X_ref, global_bundle = _compute_global_shap_bundle(
                                model_sigs=model_sigs,
                                dataset_sig=dataset_sig,
                                numeric_cols_tuple=tuple(numeric_cols),
                                max_rows=int(GLOBAL_EXPLAIN_MAX_ROWS),
                            )
                        except Exception as e:
                            st.warning(f"Global explainability failed: {e}")
                        else:
                            st.caption(
                                f"Reference: {ref_p.name} · Rows used: {X_ref.shape[0]} · "
                                f"Features: {len(numeric_cols)}"
                            )

                            scaler_g = scaler_bundle['scaler']
                            X_ref_raw = _inverse_standardize_with_sentinel(
                                np.asarray(X_ref, dtype=np.float32),
                                scaler_g,
                                SENTINEL_MISSING_VALUE,
                            )

                            tabs_g = st.tabs([f"{icon} {name}" for icon, name in zip(DISEASE_ICONS, DISEASE_NAMES)])
                            display_names = _display_feature_names(numeric_cols)
                            for tab, target, disease_name in zip(tabs_g, TARGET_COLS, DISEASE_NAMES):
                                with tab:
                                    if target not in global_bundle:
                                        st.warning(f"No global SHAP data available for {disease_name}.")
                                        continue
                                    try:
                                        shap_vals = global_bundle[target]['shap_vals']
                                        bias_vals = global_bundle[target]['bias']

                                        exp_g = shap.Explanation(
                                            values=np.asarray(shap_vals, dtype=np.float32),
                                            base_values=np.asarray(bias_vals, dtype=np.float32),
                                            data=np.asarray(X_ref_raw, dtype=np.float32),
                                            feature_names=display_names,
                                        )

                                        st.markdown("**Mean |SHAP| bar (global importance)**")
                                        plt.figure()
                                        shap.plots.bar(exp_g, max_display=int(top_k_global), show=False)
                                        fig = plt.gcf()
                                        st.pyplot(fig, use_container_width=True, clear_figure=True)

                                        st.markdown("**SHAP summary (beeswarm)**")
                                        plt.figure()
                                        shap.plots.beeswarm(exp_g, max_display=int(top_k_global), show=False)
                                        fig = plt.gcf()
                                        st.pyplot(fig, use_container_width=True, clear_figure=True)

                                        try:
                                            import pandas as pd

                                            mean_abs = np.mean(np.abs(np.asarray(shap_vals, dtype=np.float32)), axis=0)
                                            df_rank = pd.DataFrame({
                                                'Feature': display_names,
                                                'Mean |SHAP|': mean_abs,
                                            }).sort_values('Mean |SHAP|', ascending=False)
                                            df_rank.insert(0, 'Rank', np.arange(1, len(df_rank) + 1))
                                            st.markdown("**Ranked feature importance (top)**")
                                            st.dataframe(
                                                df_rank.head(int(top_k_global)),
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                        except Exception:
                                            pass
                                    except Exception as e:
                                        st.warning(f"Could not render global SHAP plot for {disease_name}: {e}")

                            with st.expander("🔁 Dependence plots — Feature interactions", expanded=False):
                                st.markdown(
                                    "Dependence plots show how a feature’s value relates to its SHAP contribution. "
                                    "Coloring by another feature helps reveal interactions (e.g., age vs glucose)."
                                )

                                dep_target = st.selectbox(
                                    "Disease model",
                                    options=TARGET_COLS,
                                    format_func=lambda t: f"{DISEASE_NAMES[TARGET_COLS.index(t)]}",
                                    key="dep_target",
                                )
                                dep_feat = st.selectbox(
                                    "Feature",
                                    options=list(numeric_cols),
                                    format_func=lambda f: FEATURE_LABELS.get(f, f),
                                    key="dep_feature",
                                )
                                dep_color = st.selectbox(
                                    "Color by (interaction feature)",
                                    options=['(auto)'] + list(numeric_cols),
                                    format_func=lambda f: 'Auto' if f == '(auto)' else FEATURE_LABELS.get(f, f),
                                    key="dep_color",
                                )
                                st.caption("Try: Age vs Fasting Glucose · BMI vs Systolic BP · BMI vs Diastolic BP")

                                if dep_target not in global_bundle:
                                    st.warning("No global SHAP data available for the selected disease model.")
                                else:
                                    try:
                                        shap_vals = np.asarray(global_bundle[dep_target]['shap_vals'], dtype=np.float32)
                                        exp_dep = shap.Explanation(
                                            values=shap_vals,
                                            data=np.asarray(X_ref_raw, dtype=np.float32),
                                            feature_names=display_names,
                                        )
                                        feat_idx = list(numeric_cols).index(dep_feat)

                                        plt.figure()
                                        if dep_color == '(auto)':
                                            shap.plots.scatter(exp_dep[:, feat_idx], show=False)
                                        else:
                                            color_idx = list(numeric_cols).index(dep_color)
                                            shap.plots.scatter(exp_dep[:, feat_idx], color=exp_dep[:, color_idx], show=False)
                                        fig = plt.gcf()
                                        st.pyplot(fig, use_container_width=True, clear_figure=True)
                                    except Exception as e:
                                        st.warning(f"Could not render dependence plot: {e}")

    # ── Optional advanced explainability ─────────────────────────────────────
    with st.expander("🧑‍🤝‍🧑 Optional — Compare Two Patients", expanded=False):
        st.markdown(
            "Save two predictions (A and B) to compare risk scores and feature drivers. "
            "This is a *model comparison tool* — not a clinical diagnosis."
        )

        if 'compare_patients' not in st.session_state:
            st.session_state['compare_patients'] = {'A': None, 'B': None}

        def _snapshot_bundle(tag: str):
            scaler_c = scaler_bundle['scaler']
            scaler_cols_c = list(scaler_bundle['numeric_cols'])
            X_c = patient_to_xgb(patient_used, numeric_cols, scaler_c, scaler_cols_c)
            shap_all = {}
            for t in TARGET_COLS:
                sv, b = _xgb_pred_contribs(xgb_models[t], X_c, numeric_cols)
                shap_all[t] = {
                    'shap_row': np.asarray(sv[0], dtype=np.float32),
                    'bias_val': float(b[0]),
                }
            return {
                'tag': tag,
                'name': pname,
                'patient': dict(patient_used),
                'results': dict(results),
                'shap': shap_all,
            }

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if st.button("Save current as Patient A", key="save_A"):
                st.session_state['compare_patients']['A'] = _snapshot_bundle('A')
                st.success("Saved as Patient A")
        with col_c2:
            if st.button("Save current as Patient B", key="save_B"):
                st.session_state['compare_patients']['B'] = _snapshot_bundle('B')
                st.success("Saved as Patient B")
        with col_c3:
            if st.button("Clear saved", key="clear_compare"):
                st.session_state['compare_patients'] = {'A': None, 'B': None}
                st.info("Cleared")

        A = st.session_state['compare_patients'].get('A')
        B = st.session_state['compare_patients'].get('B')

        if A is None or B is None:
            st.caption("Save both A and B to enable comparison.")
        else:
            try:
                import pandas as pd

                rows = []
                for target, name, icon in zip(TARGET_COLS, DISEASE_NAMES, DISEASE_ICONS):
                    pA = float(A['results'][target])
                    pB = float(B['results'][target])
                    rows.append({
                        'Disease': f"{icon} {name}",
                        'A risk (%)': round(pA * 100.0, 1),
                        'B risk (%)': round(pB * 100.0, 1),
                        'Δ (B-A) (%)': round((pB - pA) * 100.0, 1),
                    })
                st.markdown("**Risk comparison**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                sel_target = st.selectbox(
                    "Disease to compare feature drivers",
                    options=TARGET_COLS,
                    format_func=lambda t: f"{DISEASE_NAMES[TARGET_COLS.index(t)]}",
                    key="cmp_target",
                )

                shapA = np.asarray(A['shap'][sel_target]['shap_row'], dtype=np.float32)
                shapB = np.asarray(B['shap'][sel_target]['shap_row'], dtype=np.float32)
                delta = shapB - shapA

                df_d = pd.DataFrame({
                    'Feature': _display_feature_names(numeric_cols),
                    'Δ contribution (log-odds)': delta.astype(float),
                })
                df_d['|Δ|'] = df_d['Δ contribution (log-odds)'].abs()
                df_d = df_d.sort_values('|Δ|', ascending=False).head(int(top_k))
                df_d = df_d.drop(columns=['|Δ|'])
                st.markdown("**Top driver changes (B - A)**")
                st.dataframe(df_d, use_container_width=True, hide_index=True)

                st.caption(
                    "Positive Δ means the feature pushed the model *more toward higher risk* in B than in A. "
                    "Values are additive in log-odds space."
                )
            except Exception as e:
                st.warning(f"Comparison view failed: {e}")

    with st.expander("🔁 Optional — What-if Analysis (single feature)", expanded=False):
        st.markdown(
            "Explore how changing **one** feature (while keeping others fixed) changes a disease risk score. "
            "This is *not causal* and should not be interpreted as treatment guidance."
        )

        what_target = st.selectbox(
            "Disease model",
            options=TARGET_COLS,
            format_func=lambda t: f"{DISEASE_NAMES[TARGET_COLS.index(t)]}",
            key="what_target",
        )
        what_feat = st.selectbox(
            "Feature to vary",
            options=list(numeric_cols),
            format_func=lambda f: FEATURE_LABELS.get(f, f),
            key="what_feat",
        )

        cur = patient_used.get(what_feat)
        if cur is None:
            nr = NORMAL_RANGES.get(what_feat)
            if nr:
                lo, hi, _u = nr
                cur = float((lo + hi) / 2)
            else:
                cur = 0.0

        nr = NORMAL_RANGES.get(what_feat)
        if nr:
            lo, hi, unit = nr
            min_v = max(0.0, float(lo) * 0.5)
            max_v = max(float(hi) * 1.5, float(hi) + 1.0)
        else:
            unit = ''
            min_v = 0.0
            max_v = max(float(cur) * 2.0, float(cur) + 10.0, 10.0)

        rng = st.slider(
            f"Range ({unit})" if unit else "Range",
            min_value=float(min_v),
            max_value=float(max_v),
            value=(float(min_v), float(max_v)),
            step=float((max_v - min_v) / 100.0) if (max_v - min_v) > 0 else 0.1,
            key="what_range",
        )
        n_pts = st.slider("Points", min_value=10, max_value=60, value=25, step=1, key="what_points")

        try:
            import pandas as pd

            vals = np.linspace(float(rng[0]), float(rng[1]), int(n_pts), dtype=np.float32)
            probs = []
            scaler_w = scaler_bundle['scaler']
            scaler_cols_w = list(scaler_bundle['numeric_cols'])

            for v in vals:
                p_tmp = dict(patient_used)
                p_tmp[what_feat] = float(v)
                X_tmp = patient_to_xgb(p_tmp, numeric_cols, scaler_w, scaler_cols_w)
                probs.append(_predict_proba_binary(xgb_models[what_target], X_tmp))

            df_curve = pd.DataFrame({
                FEATURE_LABELS.get(what_feat, what_feat): vals.astype(float),
                'Predicted risk': np.asarray(probs, dtype=float),
            })
            st.line_chart(df_curve, x=FEATURE_LABELS.get(what_feat, what_feat), y='Predicted risk')

            cur_prob = _predict_proba_binary(xgb_models[what_target], patient_to_xgb(patient_used, numeric_cols, scaler_w, scaler_cols_w))
            st.caption(
                f"Current value: {_format_raw_value(what_feat, cur)} · Current risk: {cur_prob * 100:.1f}%"
            )
        except Exception as e:
            st.warning(f"What-if analysis failed: {e}")

    # ── AI explanation (online LLM; no local downloads) ─────────────────────
    with st.expander("🤖 AI Explanation & Guidance (Online LLM)", expanded=False):
        st.markdown(
            "Generates a patient-friendly explanation and Q&A using an **online** LLM API (no local model downloads). "
            "API keys are read from environment variables, so you don't need to store secrets in this project folder."
        )

        cfg = load_openai_compatible_config_from_env(
            default_base_url=DEFAULT_BASE_URL,
            default_model=DEFAULT_MODEL,
        )

        if cfg is None:
            st.info(
                "LLM is not configured. Set `BTP_LLM_API_KEY` (or `GROQ_API_KEY` if you're using Groq), "
                "and optionally `BTP_LLM_BASE_URL`, `BTP_LLM_MODEL`, "
                "then restart Streamlit. Until then, this section will use a rule-based fallback."
            )
            st.code(
                "# PowerShell (current session)\n"
                "$env:BTP_LLM_API_KEY = \"YOUR_KEY\"\n"
                "$env:GROQ_API_KEY = \"YOUR_KEY\"        # alternative for Groq\n"
                "$env:BTP_LLM_BASE_URL = \"https://api.groq.com/openai/v1\"   # optional\n"
                "$env:BTP_LLM_MODEL = \"llama-3.1-8b-instant\"               # optional\n\n"
                "# Persist for future terminals (CMD)\n"
                "setx BTP_LLM_API_KEY \"YOUR_KEY\"\n"
                "setx GROQ_API_KEY \"YOUR_KEY\"\n"
                "setx BTP_LLM_BASE_URL \"https://api.groq.com/openai/v1\"\n"
                "setx BTP_LLM_MODEL \"llama-3.1-8b-instant\"\n",
                language="text",
            )
        else:
            st.success(f"LLM configured: {cfg.base_url} · model={cfg.model}")

        ai_target = st.selectbox(
            "Disease model to explain",
            options=TARGET_COLS,
            format_func=lambda t: f"{DISEASE_ICONS[TARGET_COLS.index(t)]} {DISEASE_NAMES[TARGET_COLS.index(t)]}",
            key="ai_target",
        )
        ai_disease_name = DISEASE_NAMES[TARGET_COLS.index(ai_target)]
        ai_prob = float(results[ai_target])
        ai_level = risk_level(ai_prob)

        # Keep explainability cache consistent across sections.
        scaler_ai = scaler_bundle['scaler']
        scaler_cols_ai = list(scaler_bundle['numeric_cols'])
        X_ai = patient_to_xgb(patient_used, numeric_cols, scaler_ai, scaler_cols_ai)
        local_cache_key = f"{_patient_signature(patient_used, numeric_cols)}::{meta.get('artifact_dir', '')}"
        if st.session_state.get('local_shap_cache_key') != local_cache_key:
            st.session_state['local_shap_cache_key'] = local_cache_key
            st.session_state['local_shap_cache'] = {}
        local_cache = st.session_state.get('local_shap_cache', {})

        if ai_target in local_cache:
            shap_row_ai = np.asarray(local_cache[ai_target]['shap_row'], dtype=np.float32)
        else:
            shap_vals_ai, bias_ai = _xgb_pred_contribs(xgb_models[ai_target], X_ai, numeric_cols)
            shap_row_ai = np.asarray(shap_vals_ai[0], dtype=np.float32)
            local_cache[ai_target] = {
                'shap_row': shap_row_ai,
                'bias_val': float(bias_ai[0]),
            }
            st.session_state['local_shap_cache'] = local_cache

        factors_ai = build_structured_shap_factors(
            feature_names=list(numeric_cols),
            shap_row=shap_row_ai,
            patient_dict=patient_used,
            feature_labels=FEATURE_LABELS,
            normal_ranges=NORMAL_RANGES,
            top_k=6,
        )

        payload_ai = build_patient_payload_for_llm(
            disease_name=ai_disease_name,
            probability=ai_prob,
            risk_level=ai_level,
            top_increasing=factors_ai["top_increasing"],
            top_decreasing=factors_ai["top_decreasing"],
            missing_features=[FEATURE_LABELS.get(f, f) for f in missing_used],
        )

        st.caption("Context used: top SHAP factors (increasing/decreasing) + risk score.")
        with st.expander("Show LLM context (top factors)", expanded=False):
            col_ctx1, col_ctx2 = st.columns(2)
            with col_ctx1:
                st.markdown("**Increasing risk**")
                if factors_ai["top_increasing"]:
                    st.markdown("\n".join(f"- {x['label']}: {x['raw_value_display']} ({x['impact_strength']})" for x in factors_ai["top_increasing"]))
                else:
                    st.caption("No strong increasing factors detected.")
            with col_ctx2:
                st.markdown("**Decreasing risk**")
                if factors_ai["top_decreasing"]:
                    st.markdown("\n".join(f"- {x['label']}: {x['raw_value_display']} ({x['impact_strength']})" for x in factors_ai["top_decreasing"]))
                else:
                    st.caption("No strong decreasing factors detected.")

        if 'ai_explain_cache' not in st.session_state:
            st.session_state['ai_explain_cache'] = {}

        ai_cache_key = f"{_patient_signature(patient_used, numeric_cols)}::{ai_target}"
        if st.button("Generate AI explanation", key="ai_generate"):
            with st.spinner("Generating explanation..."):
                out = generate_ai_explanation(payload=payload_ai, cfg=cfg)
            st.session_state['ai_explain_cache'][ai_cache_key] = out

        out = st.session_state['ai_explain_cache'].get(ai_cache_key)
        if out is not None:
            if not out.get('ok', True):
                st.warning(f"LLM call issue: {out.get('error', 'unknown error')} (showing fallback)")
            data = out.get('data') or {}

            summary = data.get('summary')
            if summary:
                st.markdown("### Summary")
                st.write(summary)

            inc_list = data.get('key_factors_increasing')
            dec_list = data.get('key_factors_decreasing')
            if inc_list or dec_list:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Key factors increasing risk**")
                    if isinstance(inc_list, list) and inc_list:
                        st.markdown("\n".join(f"- {s}" for s in inc_list))
                    else:
                        st.caption("Not available")
                with c2:
                    st.markdown("**Key factors decreasing risk**")
                    if isinstance(dec_list, list) and dec_list:
                        st.markdown("\n".join(f"- {s}" for s in dec_list))
                    else:
                        st.caption("Not available")

            next_steps = data.get('what_to_do_next')
            if isinstance(next_steps, list) and next_steps:
                st.markdown("### What to do next")
                st.markdown("\n".join(f"- {s}" for s in next_steps))

            seek_care = data.get('when_to_seek_care')
            if isinstance(seek_care, list) and seek_care:
                st.markdown("### When to seek care")
                st.markdown("\n".join(f"- {s}" for s in seek_care))

            st.caption(f"⚠️ Disclaimer: {data.get('disclaimer') or LLM_DISCLAIMER}")

        st.markdown("---")
        st.markdown("### 💬 Ask follow-up questions")
        st.caption(
            "Questions are answered using only the model outputs + SHAP factors shown above. "
            "The chat will auto-detect kidney/liver/heart/diabetes from your question; otherwise it uses the selected model."
        )

        if st.session_state.get('ai_chat_cache_key') != ai_cache_key:
            st.session_state['ai_chat_cache_key'] = ai_cache_key
            st.session_state['ai_chat_history'] = []

        chat_hist = st.session_state.get('ai_chat_history', [])
        for msg in chat_hist:
            role = msg.get('role', 'assistant')
            content = msg.get('content', '')
            if not content:
                continue
            with st.chat_message(role):
                st.markdown(content)

        user_q = st.chat_input("Ask about kidney/liver/heart/diabetes risk and the shown factors...")
        if user_q:
            st.session_state['ai_chat_history'] = chat_hist + [{"role": "user", "content": user_q}]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    mentioned = _mentioned_disease_targets(user_q)

                    # If the user mentions multiple diseases, ask them to narrow it down.
                    if len(mentioned) > 1:
                        names = [DISEASE_NAMES[TARGET_COLS.index(t)] for t in mentioned if t in TARGET_COLS]
                        answer = (
                            "Your question mentions multiple diseases (" + ", ".join(names) + "). "
                            "Please ask about one disease at a time so I can use the correct model context. "
                            "Example: 'What is my kidney disease risk and what factors drove it?'"
                        )
                        st.markdown(answer)
                        st.caption(f"⚠️ {LLM_DISCLAIMER}")
                        reply = {"answer": answer, "disclaimer": LLM_DISCLAIMER}
                    else:
                        chat_target = mentioned[0] if len(mentioned) == 1 else ai_target
                        chat_disease_name = DISEASE_NAMES[TARGET_COLS.index(chat_target)]

                        # Build payload for the disease the user is asking about.
                        if chat_target in local_cache:
                            chat_shap_row = np.asarray(local_cache[chat_target]['shap_row'], dtype=np.float32)
                        else:
                            sv, b = _xgb_pred_contribs(xgb_models[chat_target], X_ai, numeric_cols)
                            chat_shap_row = np.asarray(sv[0], dtype=np.float32)
                            local_cache[chat_target] = {
                                'shap_row': chat_shap_row,
                                'bias_val': float(b[0]),
                            }
                            st.session_state['local_shap_cache'] = local_cache

                        chat_prob = float(results[chat_target])
                        chat_level = risk_level(chat_prob)
                        chat_factors = build_structured_shap_factors(
                            feature_names=list(numeric_cols),
                            shap_row=chat_shap_row,
                            patient_dict=patient_used,
                            feature_labels=FEATURE_LABELS,
                            normal_ranges=NORMAL_RANGES,
                            top_k=6,
                        )
                        chat_payload = build_patient_payload_for_llm(
                            disease_name=chat_disease_name,
                            probability=chat_prob,
                            risk_level=chat_level,
                            top_increasing=chat_factors["top_increasing"],
                            top_decreasing=chat_factors["top_decreasing"],
                            missing_features=[FEATURE_LABELS.get(f, f) for f in missing_used],
                        )

                        reply = generate_ai_chat_reply(
                            payload=chat_payload,
                            user_message=user_q,
                            # Avoid cross-disease confusion: each answer is grounded from scratch.
                            chat_history=None,
                            cfg=cfg,
                        )
                        answer = reply.get('answer') or "(no answer)"
                        answer = f"*Context used: {chat_disease_name} model*\n\n" + answer
                        st.markdown(answer)
                        st.caption(f"⚠️ {reply.get('disclaimer') or LLM_DISCLAIMER}")
            st.session_state['ai_chat_history'] = st.session_state['ai_chat_history'] + [
                {"role": "assistant", "content": answer}
            ]

    # ── Detailed breakdown table ───────────────────────────────────────────────
    with st.expander("📋 Detailed Results Table", expanded=True):
        import pandas as pd
        rows = []
        for target, name, icon in zip(TARGET_COLS, DISEASE_NAMES, DISEASE_ICONS):
            prob  = results[target]
            level = risk_level(prob)
            rows.append({
                'Disease':     f"{icon} {name}",
                'Probability': f"{prob:.4f}",
                'Risk (%)':    f"{prob*100:.2f}%",
                'Risk Level':  level,
                'Confidence':  '▓' * int(prob * 20) + '░' * (20 - int(prob * 20)),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Cross-disease insight ──────────────────────────────────────────────────
    st.markdown("### 🔗 Cross-Disease Insights")
    st.markdown("*These insights are simple clinical heuristics based on the 4 risk scores (the XGBoost models are trained per-disease).* ")

    insights = []
    d_risk    = results['diabetes_risk']
    h_risk    = results['heart_disease_risk']
    l_risk    = results['liver_disease_risk']
    k_risk    = results['kidney_disease_risk']

    if d_risk >= 0.5 and k_risk >= 0.35:
        insights.append(("🩸→🫘", "Diabetic Nephropathy Risk",
                         f"High diabetes risk ({d_risk*100:.0f}%) combined with kidney risk ({k_risk*100:.0f}%) suggests possible diabetic nephropathy. Consider monitoring kidney function regularly."))
    if d_risk >= 0.5 and h_risk >= 0.35:
        insights.append(("🩸→❤️", "Diabetic Cardiomyopathy Risk",
                         f"Elevated diabetes risk ({d_risk*100:.0f}%) alongside heart risk ({h_risk*100:.0f}%) may indicate diabetic cardiomyopathy. Cardiovascular monitoring recommended."))
    if k_risk >= 0.5 and h_risk >= 0.35:
        insights.append(("🫘→❤️", "Cardiorenal Syndrome",
                         f"High kidney risk ({k_risk*100:.0f}%) with elevated heart risk ({h_risk*100:.0f}%) is consistent with cardiorenal syndrome — kidney and heart disease often co-occur."))
    if l_risk >= 0.5 and d_risk >= 0.35:
        insights.append(("🫁→🩸", "NAFLD-Insulin Resistance",
                         f"Elevated liver risk ({l_risk*100:.0f}%) with diabetes risk ({d_risk*100:.0f}%) may indicate non-alcoholic fatty liver disease with insulin resistance."))
    if l_risk >= 0.5 and h_risk >= 0.35:
        insights.append(("🫁→❤️", "NAFLD-Cardiovascular Risk",
                         f"Liver risk ({l_risk*100:.0f}%) combined with heart risk ({h_risk*100:.0f}%) is associated with NAFLD-related cardiovascular complications."))

    if insights:
        for pathway, title, desc in insights:
            st.markdown(f"""
            <div style="
                background: #f0f9ff;
                border-left: 4px solid #0369a1;
                border-radius: 0 10px 10px 0;
                padding: 1rem 1.2rem;
                margin-bottom: 0.8rem;
            ">
                <strong style="color:#0369a1;">{pathway} {title}</strong><br>
                <span style="color:#334155; font-size:0.9rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            border-radius: 0 10px 10px 0;
            padding: 1rem 1.2rem;
        ">
            <strong style="color:#15803d;">✅ No significant cross-disease patterns detected.</strong><br>
            <span style="color:#334155; font-size:0.9rem;">Risk scores do not suggest comorbidity pathways. Continue routine monitoring.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Missing features note ──────────────────────────────────────────────────
    if missing_used:
        st.markdown("""
        <div class="missing-box">
        <strong>ℹ️ Missing Features Handling</strong><br>
        The following features were not provided and were auto-handled using a sentinel encoding
        (missing → -999 after scaling). The XGBoost models were trained with this convention and
        can still output risk scores with partial lab data:<br><br>
        """ + " · ".join(f"<code>{FEATURE_LABELS.get(f,f)}</code>" for f in missing_used) + """
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("")
    st.caption("⚠️ **Disclaimer:** This tool is for research purposes only. It is not a substitute for professional medical diagnosis or advice. Always consult a qualified healthcare provider.")