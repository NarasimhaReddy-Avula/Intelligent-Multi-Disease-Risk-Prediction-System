"""Shared inference engine for the Multi-Disease Risk Prediction system.

Single source of truth used by:
- webapp/app.py          (Streamlit frontend)
- api/main.py            (FastAPI backend)
- src/report/pdf_report  (PDF report generation)

Deployed model: per-disease XGBoost classifiers trained on the unified
24-feature clinical schema with sentinel (-999) missing-value encoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from src.explainability.shap_payload import build_structured_shap_factors

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_SUBDIR = Path("week7_artifacts")
SCALER_FILENAME = "scaler.pkl"
SUMMARY_FILENAME = "summary.json"

XGB_MODEL_FILES = {
    "diabetes_risk": "diabetes_XGBoost.pkl",
    "heart_disease_risk": "heart_XGBoost.pkl",
    "liver_disease_risk": "liver_XGBoost.pkl",
    "kidney_disease_risk": "kidney_XGBoost.pkl",
}

SENTINEL_MISSING_VALUE = -999.0

TARGET_COLS = ["diabetes_risk", "heart_disease_risk", "liver_disease_risk", "kidney_disease_risk"]
DISEASE_NAMES = ["Diabetes", "Heart Disease", "Liver Disease", "Kidney Disease"]
DISEASE_ICONS = ["🩸", "❤️", "🫁", "🫘"]

NUMERIC_COLS = [
    "age", "gender", "bmi",
    "systolic_bp", "diastolic_bp",
    "cholesterol_total", "glucose_fasting", "hba1c",
    "hdl", "ldl", "triglycerides",
    "alt", "ast", "alp", "albumin", "total_protein",
    "bilirubin_total", "bilirubin_direct", "ag_ratio",
    "serum_creatinine", "bun", "gfr", "protein_urine", "hemoglobin",
]

FEATURE_LABELS = {
    "age":              "Age (years)",
    "gender":           "Gender",
    "bmi":              "BMI (kg/m²)",
    "systolic_bp":      "Systolic BP (mmHg)",
    "diastolic_bp":     "Diastolic BP (mmHg)",
    "cholesterol_total":"Total Cholesterol (mg/dL)",
    "glucose_fasting":  "Fasting Glucose (mg/dL)",
    "hba1c":            "HbA1c (%)",
    "hdl":              "HDL Cholesterol (mg/dL)",
    "ldl":              "LDL Cholesterol (mg/dL)",
    "triglycerides":    "Triglycerides (mg/dL)",
    "alt":              "ALT / SGPT (U/L)",
    "ast":              "AST / SGOT (U/L)",
    "alp":              "ALP — Alkaline Phosphatase (U/L)",
    "albumin":          "Albumin (g/dL)",
    "total_protein":    "Total Protein (g/dL)",
    "bilirubin_total":  "Total Bilirubin (mg/dL)",
    "bilirubin_direct": "Direct Bilirubin (mg/dL)",
    "ag_ratio":         "A/G Ratio",
    "serum_creatinine": "Serum Creatinine (mg/dL)",
    "bun":              "BUN — Blood Urea Nitrogen (mg/dL)",
    "gfr":              "GFR — Glomerular Filtration Rate (mL/min)",
    "protein_urine":    "Urine Protein (g/dL)",
    "hemoglobin":       "Hemoglobin (g/dL)",
}

FEATURE_GROUPS = {
    "Demographics": ["age", "gender", "bmi"],
    "Cardiovascular": ["systolic_bp", "diastolic_bp", "cholesterol_total", "hdl", "ldl", "triglycerides"],
    "Diabetes Markers": ["glucose_fasting", "hba1c"],
    "Liver Function": ["alt", "ast", "alp", "albumin", "total_protein", "bilirubin_total", "bilirubin_direct", "ag_ratio"],
    "Kidney Function": ["serum_creatinine", "bun", "gfr", "protein_urine", "hemoglobin"],
}

NORMAL_RANGES = {
    "age":              (0, 120, "years"),
    "bmi":              (18.5, 24.9, "kg/m²"),
    "systolic_bp":      (90, 120, "mmHg"),
    "diastolic_bp":     (60, 80, "mmHg"),
    "cholesterol_total":(0, 200, "mg/dL"),
    "glucose_fasting":  (70, 100, "mg/dL"),
    "hba1c":            (4.0, 5.6, "%"),
    "hdl":              (40, 60, "mg/dL"),
    "ldl":              (0, 100, "mg/dL"),
    "triglycerides":    (0, 150, "mg/dL"),
    "alt":              (7, 40, "U/L"),
    "ast":              (10, 40, "U/L"),
    "alp":              (44, 147, "U/L"),
    "albumin":          (3.5, 5.0, "g/dL"),
    "total_protein":    (6.0, 8.3, "g/dL"),
    "bilirubin_total":  (0.1, 1.2, "mg/dL"),
    "bilirubin_direct": (0.0, 0.3, "mg/dL"),
    "ag_ratio":         (1.0, 2.5, ""),
    "serum_creatinine": (0.6, 1.2, "mg/dL"),
    "bun":              (7, 20, "mg/dL"),
    "gfr":              (90, 120, "mL/min"),
    "protein_urine":    (0.0, 0.15, "g/dL"),
    "hemoglobin":       (12.0, 17.5, "g/dL"),
}

RISK_COLORS = {
    "LOW":    ("#22c55e", "#dcfce7"),
    "MEDIUM": ("#f59e0b", "#fef3c7"),
    "HIGH":   ("#ef4444", "#fee2e2"),
}

DISCLAIMER = (
    "This tool is for research/educational purposes only and is not a substitute "
    "for professional medical diagnosis or advice. Always consult a qualified "
    "healthcare provider."
)


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Artifacts:
    models: Dict[str, Any]
    scaler_bundle: Dict[str, Any]
    numeric_cols: List[str]
    artifact_dir: str
    mean_auc: Optional[float] = None
    aucs: Optional[Dict[str, float]] = None
    summary_path: Optional[str] = None


def _resolve_artifact_paths(project_root: Optional[Path] = None):
    root = Path(project_root) if project_root else PROJECT_ROOT
    search_roots = [root] + list(root.parents[:4])

    tried = []
    for r in search_roots:
        artifact_dir = r / ARTIFACT_SUBDIR
        scaler_path = artifact_dir / SCALER_FILENAME
        summary_path = artifact_dir / SUMMARY_FILENAME
        model_paths = {t: artifact_dir / f for t, f in XGB_MODEL_FILES.items()}
        tried.append(str(artifact_dir))
        if scaler_path.exists() and all(p.exists() for p in model_paths.values()):
            return artifact_dir, scaler_path, model_paths, (summary_path if summary_path.exists() else None)

    searched = "\n".join(f"- {p}" for p in tried)
    raise FileNotFoundError(
        "Could not find required XGBoost artifacts (scaler.pkl + 4 *_XGBoost.pkl). Searched:\n" + searched
    )


def load_artifacts(project_root: Optional[Path] = None) -> Artifacts:
    artifact_dir, scaler_path, model_paths, summary_path = _resolve_artifact_paths(project_root)

    scaler_bundle = joblib.load(scaler_path)
    if not isinstance(scaler_bundle, dict) or "scaler" not in scaler_bundle or "numeric_cols" not in scaler_bundle:
        raise ValueError("scaler.pkl must be a dict bundle with 'scaler' and 'numeric_cols' keys.")

    models = {t: joblib.load(p) for t, p in model_paths.items()}

    mean_auc = None
    aucs: Optional[Dict[str, float]] = None
    if summary_path is not None:
        try:
            data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            raw = data.get("xgb_aucs")
            if isinstance(raw, dict) and raw:
                aucs = {k: float(v) for k, v in raw.items()}
                mean_auc = float(np.mean(list(aucs.values())))
        except Exception:
            mean_auc = None

    return Artifacts(
        models=models,
        scaler_bundle=scaler_bundle,
        numeric_cols=list(scaler_bundle["numeric_cols"]),
        artifact_dir=str(artifact_dir),
        mean_auc=mean_auc,
        aucs=aucs,
        summary_path=(str(summary_path) if summary_path else None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def patient_to_xgb(patient_dict: Dict[str, Any], numeric_cols: Sequence[str], scaler, scaler_cols: Sequence[str]) -> np.ndarray:
    """Convert raw clinical values -> sentinel-encoded array for XGBoost."""
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


def predict_proba_binary(model, X: np.ndarray) -> float:
    """Return P(y=1) for a binary classifier-like model."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        if proba.ndim == 1:
            return float(proba[0])
    try:
        import xgboost as xgb
        dmat = xgb.DMatrix(X)
        pred = np.asarray(model.predict(dmat))
        return float(pred[0])
    except Exception as e:
        raise TypeError("Loaded XGBoost artifact does not support probability prediction.") from e


def predict(patient_dict: Dict[str, Any], artifacts: Artifacts) -> Dict[str, float]:
    """Run XGBoost inference -> dict of disease -> probability."""
    scaler = artifacts.scaler_bundle["scaler"]
    scaler_cols = list(artifacts.scaler_bundle["numeric_cols"])
    X = patient_to_xgb(patient_dict, artifacts.numeric_cols, scaler, scaler_cols)
    return {t: predict_proba_binary(artifacts.models[t], X) for t in TARGET_COLS}


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def risk_level(prob: float) -> str:
    if prob >= 0.65:
        return "HIGH"
    if prob >= 0.35:
        return "MEDIUM"
    return "LOW"


def shap_contributions(model, X: np.ndarray, feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """TreeSHAP per-feature contributions via XGBoost pred_contribs.

    Returns (shap_values [n x f], bias [n]). Contributions are in log-odds space.
    """
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmat = xgb.DMatrix(X, feature_names=list(feature_names))
    contribs = np.asarray(booster.predict(dmat, pred_contribs=True))
    if contribs.ndim != 2 or contribs.shape[0] != X.shape[0] or contribs.shape[1] != (X.shape[1] + 1):
        raise ValueError(
            f"Unexpected pred_contribs shape {tuple(contribs.shape)}; expected ({X.shape[0]}, {X.shape[1] + 1})."
        )
    return contribs[:, :-1], contribs[:, -1]


def structured_factors(
    target: str,
    patient_dict: Dict[str, Any],
    artifacts: Artifacts,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Per-disease SHAP factors in structured form (for UI/PDF/LLM)."""
    scaler = artifacts.scaler_bundle["scaler"]
    scaler_cols = list(artifacts.scaler_bundle["numeric_cols"])
    X = patient_to_xgb(patient_dict, artifacts.numeric_cols, scaler, scaler_cols)
    shap_vals, bias = shap_contributions(artifacts.models[target], X, artifacts.numeric_cols)
    shap_row = np.asarray(shap_vals[0], dtype=np.float32)
    factors = build_structured_shap_factors(
        feature_names=list(artifacts.numeric_cols),
        shap_row=shap_row,
        patient_dict=patient_dict,
        feature_labels=FEATURE_LABELS,
        normal_ranges=NORMAL_RANGES,
        top_k=top_k,
    )
    factors["bias"] = float(bias[0])
    factors["shap_row"] = shap_row
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def inverse_standardize_with_sentinel(X_scaled: np.ndarray, scaler, sentinel: float) -> np.ndarray:
    X = np.asarray(X_scaled, dtype=np.float32)
    mask = X == float(sentinel)
    X2 = X.copy()
    X2[mask] = 0.0
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is not None and scale is not None:
        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        if mean.shape[0] == X.shape[1] and scale.shape[0] == X.shape[1]:
            X_raw = (X2 * scale) + mean
            X_raw[mask] = np.nan
            return np.asarray(X_raw, dtype=np.float32)
    try:
        X_raw = np.asarray(scaler.inverse_transform(X2), dtype=np.float32)
        X_raw[mask] = np.nan
        return X_raw
    except Exception:
        X_disp = X.astype(np.float32)
        X_disp[mask] = np.nan
        return X_disp


def format_raw_value(feat: str, raw_val) -> str:
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return "not provided"
    if feat == "gender":
        return "Male" if float(raw_val) >= 0.5 else "Female"
    nr = NORMAL_RANGES.get(feat)
    unit = nr[2] if nr else ""
    try:
        v = float(raw_val)
        if unit:
            return f"{v:.2f} {unit}" if abs(v) < 100 else f"{v:.1f} {unit}"
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"
    except Exception:
        return str(raw_val)


def feature_level_phrase(feat: str, raw_val) -> str:
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


def provided_and_missing(patient_dict: Dict[str, Any], numeric_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    provided = [f for f in numeric_cols if patient_dict.get(f) is not None]
    missing = [f for f in numeric_cols if patient_dict.get(f) is None]
    return provided, missing


RISK_COLORS = {
    "LOW": ("#22c55e", "#dcfce7"),
    "MEDIUM": ("#f59e0b", "#fef3c7"),
    "HIGH": ("#ef4444", "#fee2e2"),
    "INSUFFICIENT": ("#94a3b8", "#f1f5f9"),
}


# Per-disease lab panels. Each model was trained only on its own cohort's rows
# (~2K-100K samples), so disease-specific markers are REQUIRED for a meaningful
# score: unseen -999 sentinels are out-of-distribution for that model and the
# extrapolated probability is not trustworthy.
DISEASE_PANELS: Dict[str, List[str]] = {
    "diabetes_risk": ["glucose_fasting", "hba1c"],
    "heart_disease_risk": [
        "systolic_bp", "diastolic_bp", "cholesterol_total", "hdl", "ldl", "triglycerides",
    ],
    "liver_disease_risk": [
        "alt", "ast", "alp", "bilirubin_total", "bilirubin_direct",
        "albumin", "total_protein", "ag_ratio",
    ],
    "kidney_disease_risk": [
        "serum_creatinine", "bun", "gfr", "protein_urine", "hemoglobin",
    ],
}


def panel_provided(patient_dict: Dict[str, Any], target: str) -> List[str]:
    """Disease-panel features the patient actually provided."""
    return [f for f in DISEASE_PANELS.get(target, []) if patient_dict.get(f) is not None]


def panel_hint(target: str) -> str:
    """Short comma-separated marker names for a disease panel, e.g. 'Serum Creatinine, BUN, GFR'."""
    return ", ".join(
        FEATURE_LABELS.get(f, f).split(" (")[0].split(" — ")[0]
        for f in DISEASE_PANELS.get(target, [])
    )


def disease_readiness(patient_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per-disease: whether enough disease-specific markers exist to trust the score."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in TARGET_COLS:
        provided = panel_provided(patient_dict, t)
        out[t] = {
            "panel": list(DISEASE_PANELS.get(t, [])),
            "provided": provided,
            "scoreable": len(provided) > 0,
        }
    return out


# Feature -> disease family it clinically belongs to (for cross-disease attribution).
_FEATURE_FAMILY: Dict[str, str] = {
    f: group for group, feats in FEATURE_GROUPS.items() for f in feats
}


def panel_coverage(patient_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per-disease marker coverage: how many of its panel markers were provided."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in TARGET_COLS:
        panel = DISEASE_PANELS.get(t, [])
        provided = panel_provided(patient_dict, t)
        out[t] = {
            "provided": len(provided),
            "total": len(panel),
            "pct": (len(provided) / len(panel)) if panel else 1.0,
        }
    return out


def cross_disease_attribution(
    target: str,
    patient_dict: Dict[str, Any],
    artifacts: Artifacts,
) -> Dict[str, float]:
    """SHAP contributions for one disease, summed by feature family.

    Shows which *other* disease families drive this score — e.g. how much of a
    kidney score comes from diabetes markers vs kidney markers vs demographics.
    Returns {family_name: shap_sum_in_log_odds}.
    """
    scaler = artifacts.scaler_bundle["scaler"]
    scaler_cols = list(artifacts.scaler_bundle["numeric_cols"])
    X = patient_to_xgb(patient_dict, artifacts.numeric_cols, scaler, scaler_cols)
    shap_vals, _bias = shap_contributions(artifacts.models[target], X, artifacts.numeric_cols)
    row = np.asarray(shap_vals[0], dtype=np.float64)
    out: Dict[str, float] = {}
    for feat, val in zip(artifacts.numeric_cols, row):
        fam = _FEATURE_FAMILY.get(feat, "Other")
        out[fam] = out.get(fam, 0.0) + float(val)
    return out


def panel_ablation(
    target: str,
    patient_dict: Dict[str, Any],
    artifacts: Artifacts,
) -> Dict[str, float]:
    """Score one disease three ways: full input vs its panel only vs panel removed.

    Quantifies the disease panel's own contribution vs cross-disease signal:
    - panel_only: demographics + this disease's markers (other diseases' markers stripped)
    - panel_removed: everything except this disease's markers
    """
    shared = {f: patient_dict.get(f) for f in ("age", "gender", "bmi")
              if patient_dict.get(f) is not None}
    panel = set(DISEASE_PANELS.get(target, []))

    panel_only = dict(shared)
    for f in panel:
        if patient_dict.get(f) is not None:
            panel_only[f] = patient_dict[f]

    panel_removed = {f: v for f, v in patient_dict.items() if f not in panel}

    full = predict(patient_dict, artifacts)[target]
    return {
        "full": full,
        "panel_only": predict(panel_only, artifacts)[target],
        "panel_removed": predict(panel_removed, artifacts)[target],
    }
