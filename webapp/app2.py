"""
⚠️ EXPERIMENTAL COMPARISON BUILD — DO NOT DEMO ⚠️

This Streamlit variant drives the FT-Transformer checkpoint, whose training
runs were inconsistent. The deployed system is `webapp/app.py` (XGBoost).
Kept for research comparison only.

Run with: streamlit run webapp/app2.py   (requires torch + full requirements.txt)
"""

import sys
import os
from pathlib import Path

# ── Project root resolution ───────────────────────────────────────────────────
# webapp/app.py is one level inside the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import joblib
import streamlit as st
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACT_SUBDIR = Path('week7_artifacts')
CKPT_FILENAME   = 'best_model.pt'
SCALER_FILENAME = 'scaler.pkl'
SUMMARY_FILENAME = 'summary.json'
DEVICE      = torch.device('cpu')   # CPU is fine for single inference

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
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class MultiOutputFTTransformer(nn.Module):
    def __init__(self, n_features, n_outputs=4, d_token=64, n_heads=8, n_layers=4,
                 ff_mult=4, attn_dropout=0.1, ff_dropout=0.1, out_dropout=0.1):
        super().__init__()
        self.n_outputs      = n_outputs
        self.feature_weight = nn.Parameter(torch.empty(n_features, d_token))
        self.feature_bias   = nn.Parameter(torch.empty(n_features, d_token))
        self.disease_tokens = nn.Parameter(torch.empty(1, n_outputs, d_token))
        enc = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ff_mult,
            dropout=ff_dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.token_drop = nn.Dropout(attn_dropout)
        self.final_norm = nn.LayerNorm(d_token)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_token), nn.Dropout(out_dropout), nn.Linear(d_token, 1))
            for _ in range(n_outputs)
        ])

    def tokenize(self, x):
        return x.unsqueeze(-1) * self.feature_weight + self.feature_bias

    def forward(self, x):
        B = x.size(0)
        tokens  = torch.cat([self.disease_tokens.expand(B, -1, -1), self.tokenize(x)], dim=1)
        encoded = self.final_norm(self.encoder(self.token_drop(tokens)))
        dr      = encoded[:, :self.n_outputs, :]
        return torch.cat([h(dr[:, i, :]) for i, h in enumerate(self.heads)], dim=1)


def resolve_artifact_paths():
    """Find FT-Transformer artifacts even when the app is launched from nested copies."""
    search_roots = [PROJECT_ROOT]
    # Covers layouts like <workspace>/tmp/gh_repo/webapp/app.py
    search_roots.extend(PROJECT_ROOT.parents[:4])

    tried_dirs = []
    for root in search_roots:
        artifact_dir = root / ARTIFACT_SUBDIR
        ckpt_path = artifact_dir / CKPT_FILENAME
        scaler_path = artifact_dir / SCALER_FILENAME
        summary_path = artifact_dir / SUMMARY_FILENAME
        tried_dirs.append(str(artifact_dir))
        if ckpt_path.exists() and scaler_path.exists():
            return ckpt_path, scaler_path, (summary_path if summary_path.exists() else None)

    searched = "\n".join(f"- {p}" for p in tried_dirs)
    raise FileNotFoundError(
        "Could not find required FT-Transformer artifacts best_model.pt and scaler.pkl. "
        f"Searched:\n{searched}"
    )


def _safe_mean_auc_from_summary(summary_path):
    if summary_path is None:
        return None
    try:
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        return float(data.get('best_mean_val_auc', 0)) if data.get('best_mean_val_auc') is not None else None
    except Exception:
        return None


def load_checkpoint_compat(path):
    """Load checkpoints across torch versions (2.6+ defaults to weights_only=True)."""
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        # Older torch versions do not expose weights_only.
        return torch.load(path, map_location=DEVICE)


@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler once, cache for session."""
    ckpt_path, scaler_path, summary_path = resolve_artifact_paths()

    scaler_bundle = joblib.load(scaler_path)
    scaler = scaler_bundle['scaler']
    scaler_cols = list(scaler_bundle['numeric_cols'])

    ckpt      = load_checkpoint_compat(ckpt_path)
    if not isinstance(ckpt, dict):
        raise TypeError('Checkpoint format is invalid: expected dict.')

    feature_cols = ckpt.get('feature_cols')
    if feature_cols is None:
        # Backward-compatible fallback for older checkpoints.
        feature_cols = scaler_cols + [f'{c}_mask' for c in scaler_cols]
    else:
        feature_cols = list(feature_cols)

    model_cfg = dict(ckpt.get('model_cfg', {}))
    model_cfg.pop('n_features', None)
    model_cfg.pop('n_outputs', None)

    state_dict = ckpt.get('model_state_dict') or ckpt.get('state_dict')
    if state_dict is None:
        raise KeyError("Checkpoint is missing 'model_state_dict' or 'state_dict'.")

    expected_n_features = state_dict['feature_weight'].shape[0]
    if len(feature_cols) != expected_n_features:
        raise ValueError(
            'Checkpoint feature order mismatch: '
            f'feature_cols has {len(feature_cols)} entries but model expects {expected_n_features}.'
        )

    model = MultiOutputFTTransformer(
        n_features=expected_n_features, n_outputs=4, **model_cfg
    )
    model.load_state_dict(state_dict)
    model.eval()
    meta = {
        'artifact_dir': str(ckpt_path.parent),
        'mean_auc': _safe_mean_auc_from_summary(summary_path),
    }
    return model, scaler_bundle, feature_cols, ckpt, meta


def build_feature_vector(patient_dict, scaler_bundle, feature_cols):
    """
    Convert raw patient dict -> model-ready vector in checkpoint feature order.

    For each numeric feature:
      - Present value: scaled value, mask=1
      - Missing value: scaler mean (which transforms to ~0), mask=0

    Returns: np.array shape (1, len(feature_cols))
    """
    scaler = scaler_bundle['scaler']
    scaler_cols = list(scaler_bundle['numeric_cols'])

    row = {c: 0.0 for c in feature_cols}

    raw = np.array([[
        float(patient_dict[c]) if (c in patient_dict and patient_dict[c] is not None) else np.nan
        for c in scaler_cols
    ]], dtype=np.float32)

    # Replace missing numeric values with scaler means before transform.
    for i in range(len(scaler_cols)):
        if np.isnan(raw[0, i]):
            raw[0, i] = scaler.mean_[i]

    scaled = scaler.transform(raw)[0]

    for i, col in enumerate(scaler_cols):
        if col in row:
            row[col] = float(scaled[i])

    for col, val in patient_dict.items():
        mask_col = f'{col}_mask'
        if mask_col in row:
            row[mask_col] = 1.0 if val is not None else 0.0

    return np.array([[row[c] for c in feature_cols]], dtype=np.float32)


def predict(patient_dict, model, scaler_bundle, feature_cols):
    """Run inference and return dict of disease → probability."""
    X = build_feature_vector(patient_dict, scaler_bundle, feature_cols)
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]
    return {target: float(probs[i]) for i, target in enumerate(TARGET_COLS)}


def risk_level(prob):
    if prob >= 0.65:   return 'HIGH'
    elif prob >= 0.35: return 'MEDIUM'
    else:              return 'LOW'


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
    model, scaler_bundle, feature_cols, ckpt, meta = load_model_and_scaler()
    model_loaded = True
    model_epoch  = ckpt.get('epoch', '?')
    model_auc    = meta.get('mean_auc') if meta.get('mean_auc') is not None else ckpt.get('mean_val_auc', 0)
except Exception as e:
    model_loaded = False
    model_error  = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MedRisk AI</h1>
    <p>Intelligent Multi-Disease Risk Prediction Platform</p>
    <span class="badge">FT-Transformer comparison build · Week 7 artifacts · IIIT Sri City BTP 2025–26</span>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model could not be loaded: {model_error}")
    st.info("Expected artifacts: week7_artifacts/best_model.pt and week7_artifacts/scaler.pkl")
    st.stop()

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Model", "FT-Transformer")
with col_info2:
    st.metric("Best Epoch", model_epoch)
with col_info3:
    st.metric("Val Mean AUC", f"{model_auc:.4f}" if isinstance(model_auc, (int, float)) else "n/a")

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
provided = [f for f in NUMERIC_COLS if patient_dict.get(f) is not None]
missing  = [f for f in NUMERIC_COLS if patient_dict.get(f) is None]

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
predict_clicked = st.button("🔍 Predict Disease Risk", type="primary")

if predict_clicked:
    with st.spinner("Running FT-Transformer inference on Week 7 artifacts..."):
        try:
            results = predict(patient_dict, model, scaler_bundle, feature_cols)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    pname = patient_name if patient_name.strip() else "Patient"

    st.markdown("---")
    st.markdown(f"## 📊 Risk Assessment — {pname}")
    st.markdown(f"*{len(provided)} features provided · {len(missing)} auto-handled*")
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
    st.markdown("*This FT-Transformer frontend is the comparison baseline; use it side-by-side with the XGBoost frontend to compare risk outputs.*")

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
    if missing:
        st.markdown("""
        <div class="missing-box">
        <strong>ℹ️ Missing Features Handling</strong><br>
        The following features were not provided and were auto-handled using the FT-Transformer's
        masking mechanism (mask=0, value=scaled mean). The model was trained on partially observed
        data and accounts for this automatically:<br><br>
        """ + " · ".join(f"<code>{FEATURE_LABELS.get(f,f)}</code>" for f in missing) + """
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("")
    st.caption("⚠️ **Disclaimer:** This tool is for research purposes only. It is not a substitute for professional medical diagnosis or advice. Always consult a qualified healthcare provider.")