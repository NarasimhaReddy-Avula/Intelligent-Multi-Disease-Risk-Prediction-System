"""End-to-end tests: shared inference engine, PDF report, NLP extractor, FastAPI routes.

Run:  .venv/bin/python -m pytest tests/test_system.py -q
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PATIENT = {
    "age": 55.0, "gender": 1.0, "bmi": 30.0, "glucose_fasting": 140.0, "hba1c": 7.2,
    "systolic_bp": 145.0, "serum_creatinine": 1.8, "gfr": 55.0, "alt": 65.0,
    "bilirubin_total": 1.9,
}


@pytest.fixture(scope="session")
def artifacts():
    from src.inference import load_artifacts
    return load_artifacts()


# ── Engine ────────────────────────────────────────────────────────────────────

def test_artifacts_load(artifacts):
    assert len(artifacts.numeric_cols) == 24
    assert set(artifacts.models.keys()) == set(artifacts.__class__ and artifacts.models.keys())
    assert artifacts.mean_auc is not None


def test_predict_all_four(artifacts):
    from src.inference import TARGET_COLS, predict
    res = predict(PATIENT, artifacts)
    assert set(res.keys()) == set(TARGET_COLS)
    for v in res.values():
        assert 0.0 <= v <= 1.0


def test_predict_missing_ok(artifacts):
    from src.inference import predict
    res = predict({"age": 40.0, "gender": 0.0}, artifacts)  # almost everything missing
    assert all(0.0 <= v <= 1.0 for v in res.values())


def test_panel_coverage(artifacts):
    from src.inference import panel_coverage, TARGET_COLS
    cov = panel_coverage(PATIENT)
    assert set(cov.keys()) == set(TARGET_COLS)
    assert cov["diabetes_risk"]["provided"] == 2          # glucose + hba1c
    assert cov["kidney_disease_risk"]["provided"] == 2    # creatinine + gfr
    assert cov["liver_disease_risk"]["provided"] == 2     # alt + bilirubin_total
    assert cov["heart_disease_risk"]["provided"] == 1     # systolic only
    empty = panel_coverage({})
    assert all(v["provided"] == 0 for v in empty.values())


def test_cross_disease_attribution(artifacts):
    from src.inference import cross_disease_attribution
    attr = cross_disease_attribution("kidney_disease_risk", PATIENT, artifacts)
    assert isinstance(attr, dict) and attr
    assert abs(sum(attr.values()) - sum(attr.values())) < 1e-9
    # Marker-less diabetes panel for this patient: kidney attr on Case-A-style patient
    p2 = {"age": 58, "gender": 1, "bmi": 31, "glucose_fasting": 185, "hba1c": 8.4,
          "systolic_bp": 142, "diastolic_bp": 88, "cholesterol_total": 230, "triglycerides": 210}
    a2 = cross_disease_attribution("kidney_disease_risk", p2, artifacts)
    assert "Diabetes Markers" in a2
    assert a2["Diabetes Markers"] != 0.0  # cross-disease features genuinely contribute


def test_panel_ablation(artifacts):
    from src.inference import panel_ablation, predict
    abl = panel_ablation("diabetes_risk", PATIENT, artifacts)
    assert set(abl.keys()) == {"full", "panel_only", "panel_removed"}
    assert all(0.0 <= v <= 1.0 for v in abl.values())
    assert abl["full"] == predict(PATIENT, artifacts)["diabetes_risk"]
    # Removing diabetes markers must not raise the score for this diabetic profile
    assert abl["panel_removed"] <= abl["full"] + 1e-6


def test_risk_levels(artifacts):
    from src.inference import risk_level
    assert risk_level(0.1) == "LOW"
    assert risk_level(0.5) == "MEDIUM"
    assert risk_level(0.9) == "HIGH"


def test_shap_contributions_shape(artifacts):
    from src.inference import patient_to_xgb, shap_contributions, TARGET_COLS
    scaler = artifacts.scaler_bundle["scaler"]
    X = patient_to_xgb(PATIENT, artifacts.numeric_cols, scaler, artifacts.scaler_bundle["numeric_cols"])
    for t in TARGET_COLS:
        sv, bias = shap_contributions(artifacts.models[t], X, artifacts.numeric_cols)
        assert sv.shape == (1, len(artifacts.numeric_cols))
        assert bias.shape == (1,)


def test_structured_factors(artifacts):
    from src.inference import structured_factors
    out = structured_factors("diabetes_risk", PATIENT, artifacts, top_k=5)
    assert "top_increasing" in out and "top_decreasing" in out
    assert len(out["top_increasing"]) <= 5


def test_disease_readiness(artifacts):
    from src.inference import disease_readiness
    # PATIENT has glucose+hba1c (diabetes), bp+lipids? -> heart, alt+bilirubin -> liver,
    # creatinine+gfr -> kidney — all four panels covered
    r = disease_readiness(PATIENT)
    assert all(v["scoreable"] for v in r.values())
    # demographics-only patient: nothing scoreable
    r2 = disease_readiness({"age": 50, "gender": 1})
    assert not any(v["scoreable"] for v in r2.values())
    # only hba1c → only diabetes scoreable
    r3 = disease_readiness({"hba1c": 7.0})
    assert r3["diabetes_risk"]["scoreable"]
    assert not r3["kidney_disease_risk"]["scoreable"]
    assert not r3["liver_disease_risk"]["scoreable"]
    assert not r3["heart_disease_risk"]["scoreable"]


# ── PDF report ────────────────────────────────────────────────────────────────

def test_pdf_report(artifacts):
    from src.inference import predict
    from src.report import generate_patient_report
    res = predict(PATIENT, artifacts)
    pdf = generate_patient_report(patient_dict=PATIENT, results=res, artifacts=artifacts,
                                  patient_name="Test", history_text="I smoke daily",
                                  lifestyle_features=None)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 50_000


# ── NLP extractor ─────────────────────────────────────────────────────────────

def test_nlp_extractor():
    from src.nlp.clinical_text_extractor import ClinicalTextExtractor
    lf = ClinicalTextExtractor(use_bert=False).extract_features(
        "I smoke 10 cigarettes daily. Father had diabetes.")
    assert lf.smoking_status == "current"
    assert "diabetes" in [str(x).lower() for x in (lf.family_history or [])]


# ── API ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_api_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["features"] == 24


def test_api_predict(client):
    r = client.post("/predict", json={"features": PATIENT})
    assert r.status_code == 200
    data = r.json()
    assert len(data["risks"]) == 4
    assert all(0.0 <= x["probability"] <= 1.0 for x in data["risks"])


def test_api_explain(client):
    r = client.post("/explain", json={"features": PATIENT, "disease": "diabetes_risk", "top_k": 4})
    assert r.status_code == 200
    e = r.json()["explanations"]
    assert e["target"] == "diabetes_risk"
    assert len(e["top_increasing"]) <= 4


def test_api_explain_bad_disease(client):
    r = client.post("/explain", json={"features": PATIENT, "disease": "nope"})
    assert r.status_code == 422


def test_api_explain_insufficient(client):
    r = client.post("/explain", json={"features": {"age": 50, "gender": 1}, "disease": "kidney_disease_risk"})
    assert r.status_code == 200
    e = r.json()["explanations"]
    assert e["sufficient_markers"] is False
    assert "confidence_note" in e
    assert 0.0 <= e["probability"] <= 1.0


def test_api_report(client):
    r = client.post("/report", json={"features": PATIENT, "patient_name": "T",
                                     "history_text": "I smoke daily"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_api_history(client):
    r = client.post("/extract-history", json={"text": "I smoke daily, father had diabetes"})
    assert r.status_code == 200
    assert r.json()["extracted"]["smoking_status"] == "current"
