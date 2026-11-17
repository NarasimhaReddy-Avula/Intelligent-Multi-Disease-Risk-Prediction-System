"""
FastAPI Application Entry Point
Multi-Disease AI Healthcare Platform API

Endpoints:
    GET  /health        — liveness + loaded-artifact summary
    POST /predict       — 4-disease risk probabilities + risk levels
    POST /explain       — per-disease SHAP drivers for a patient
    POST /report        — full multi-disease PDF patient report (application/pdf)
    POST /extract-history — NLP lifestyle-factor extraction from free text

Deployed model: per-disease XGBoost classifiers (sentinel -999 encoding),
shared with the Streamlit app via src.inference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.inference import (
    Artifacts,
    DISEASE_NAMES,
    DISEASE_PANELS,
    FEATURE_LABELS,
    NORMAL_RANGES,
    TARGET_COLS,
    disease_readiness,
    format_raw_value,
    load_artifacts,
    panel_coverage,
    panel_hint,
    patient_to_xgb,
    predict,
    provided_and_missing,
    risk_level,
    shap_contributions,
    structured_factors,
)
from src.explainability.shap_payload import build_structured_shap_factors
from src.report import generate_patient_report

try:
    from src.nlp.clinical_text_extractor import ClinicalTextExtractor
    _NLP_OK = True
except Exception:
    _NLP_OK = False

app = FastAPI(
    title="MedRisk AI — Multi-Disease Risk Prediction API",
    description=(
        "Predicts risk for Diabetes, Heart Disease, Liver Disease and Kidney Disease "
        "from a 24-feature clinical schema. XGBoost models with sentinel (-999) "
        "missing-value encoding; TreeSHAP explanations; PDF patient reports. "
        "For research/educational use only — not a medical device."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifacts: Optional[Artifacts] = None


def get_artifacts() -> Artifacts:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts()
    return _artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """Raw clinical features; omitted fields are treated as missing (sentinel -999)."""
    features: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Feature name → value. Unknown keys are ignored; missing keys are auto-handled.",
    )

    def patient_dict(self, numeric_cols: List[str]) -> Dict[str, Any]:
        return {f: self.features.get(f) for f in numeric_cols}


class DiseaseRisk(BaseModel):
    disease: str
    target: str
    probability: float
    risk_level: str
    sufficient_markers: bool = True
    panel_provided: List[str] = Field(default_factory=list)
    markers_provided: int = 0
    markers_total: int = 0


class PredictResponse(BaseModel):
    risks: List[DiseaseRisk]
    provided_features: List[str]
    missing_features: List[str]
    disclaimer: str


class ExplainRequest(PatientInput):
    disease: Optional[str] = Field(
        default=None,
        description="Target: diabetes_risk|heart_disease_risk|liver_disease_risk|kidney_disease_risk. "
                    "If omitted, explanations for all four diseases are returned.")
    top_k: int = Field(default=8, ge=1, le=24)


class ReportRequest(PatientInput):
    patient_name: str = "Patient"
    history_text: Optional[str] = Field(
        default=None, description="Free-text patient history; lifestyle factors are extracted for the report.")


class HistoryRequest(BaseModel):
    text: str = Field(..., min_length=3)


DISCLAIMER = (
    "For research/educational purposes only — not a medical diagnosis. "
    "Always consult a qualified healthcare provider."
)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "MedRisk AI API", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health_check():
    art = get_artifacts()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model": "XGBoost (per-disease, sentinel encoding)",
        "diseases": list(DISEASE_NAMES),
        "features": len(art.numeric_cols),
        "mean_auc": art.mean_auc,
    }


@app.get("/features")
async def list_features():
    art = get_artifacts()
    return {
        f: {
            "label": FEATURE_LABELS.get(f, f),
            "normal_range": list(NORMAL_RANGES[f]) if f in NORMAL_RANGES else None,
        }
        for f in art.numeric_cols
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(body: PatientInput):
    art = get_artifacts()
    patient = body.patient_dict(list(art.numeric_cols))
    try:
        results = predict(patient, art)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    provided, missing = provided_and_missing(patient, list(art.numeric_cols))
    readiness = disease_readiness(patient)
    cov = panel_coverage(patient)
    return PredictResponse(
        risks=[
            DiseaseRisk(
                disease=DISEASE_NAMES[TARGET_COLS.index(t)],
                target=t,
                probability=results[t],
                risk_level=risk_level(results[t]),
                sufficient_markers=readiness[t]["scoreable"],
                panel_provided=readiness[t]["provided"],
                markers_provided=cov[t]["provided"],
                markers_total=cov[t]["total"],
            )
            for t in TARGET_COLS
        ],
        provided_features=provided,
        missing_features=missing,
        disclaimer=DISCLAIMER,
    )


def _explain_one(target: str, patient: Dict[str, Any], art: Artifacts, top_k: int) -> Dict[str, Any]:
    scoreable = disease_readiness(patient)[target]["scoreable"]
    numeric_cols = list(art.numeric_cols)
    scaler = art.scaler_bundle["scaler"]
    scaler_cols = list(art.scaler_bundle["numeric_cols"])
    X = patient_to_xgb(patient, numeric_cols, scaler, scaler_cols)
    sv, bias = shap_contributions(art.models[target], X, numeric_cols)
    factors = build_structured_shap_factors(
        feature_names=numeric_cols,
        shap_row=sv[0],
        patient_dict=patient,
        feature_labels=FEATURE_LABELS,
        normal_ranges=NORMAL_RANGES,
        top_k=top_k,
    )
    prob = predict(patient, art)[target]
    out = {
        "disease": DISEASE_NAMES[TARGET_COLS.index(target)],
        "target": target,
        "probability": prob,
        "risk_level": risk_level(prob),
        "sufficient_markers": scoreable,
        "base_value_log_odds": float(bias[0]),
        "top_increasing": factors["top_increasing"],
        "top_decreasing": factors["top_decreasing"],
    }
    if not scoreable:
        out["confidence_note"] = (
            "Cross-disease estimate only — no disease-specific markers provided. "
            f"For better accuracy, provide at least one of: {panel_hint(target)}."
        )
    return out


@app.post("/explain")
async def explain_endpoint(body: ExplainRequest):
    art = get_artifacts()
    patient = body.patient_dict(list(art.numeric_cols))
    if body.disease is not None and body.disease not in TARGET_COLS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown disease target '{body.disease}'. Valid: {TARGET_COLS}",
        )
    targets = [body.disease] if body.disease else list(TARGET_COLS)
    try:
        out = [_explain_one(t, patient, art, body.top_k) for t in targets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")
    return {"explanations": out if body.disease is None else out[0], "disclaimer": DISCLAIMER}


@app.post("/report")
async def report_endpoint(body: ReportRequest):
    art = get_artifacts()
    patient = body.patient_dict(list(art.numeric_cols))
    try:
        results = predict(patient, art)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    lifestyle = None
    if body.history_text and _NLP_OK:
        try:
            lifestyle = ClinicalTextExtractor(use_bert=False).extract_features(body.history_text)
        except Exception:
            lifestyle = None

    try:
        pdf_bytes = generate_patient_report(
            patient_dict=patient,
            results=results,
            artifacts=art,
            patient_name=body.patient_name or "Patient",
            history_text=body.history_text,
            lifestyle_features=lifestyle,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    fname = f"medrisk_report_{(body.patient_name or 'patient').replace(' ', '_').lower()}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/extract-history")
async def extract_history_endpoint(body: HistoryRequest):
    if not _NLP_OK:
        raise HTTPException(status_code=503, detail="NLP extractor unavailable in this environment.")
    try:
        lf = ClinicalTextExtractor(use_bert=False).extract_features(body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    return {
        "extracted": {
            "smoking_status": lf.smoking_status,
            "smoking_frequency": lf.smoking_frequency,
            "alcohol_status": lf.alcohol_status,
            "alcohol_frequency": lf.alcohol_frequency,
            "physical_activity": lf.physical_activity,
            "family_history": lf.family_history,
            "symptoms": lf.symptoms,
            "comorbidities": getattr(lf, "comorbidities", None),
            "medications": getattr(lf, "medications", None),
        },
        "note": "Extracted factors enrich explanations/reports; they do not change model inputs.",
        "disclaimer": DISCLAIMER,
    }
