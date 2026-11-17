from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


RiskLevel = Literal["Low", "Moderate", "High", "Unknown"]


def _normalize_disease_name(disease: str) -> str:
    """Map incoming disease names to canonical keys used in the app."""

    d = (disease or "").strip()
    if not d:
        return ""

    d_norm = d.lower().strip()
    aliases = {
        "diabetes": "Diabetes",
        "heart": "Heart Disease",
        "heart disease": "Heart Disease",
        "cardiac": "Heart Disease",
        "liver": "Liver Disease",
        "liver disease": "Liver Disease",
        "kidney": "Kidney Disease",
        "kidney disease": "Kidney Disease",
        "chronic kidney disease": "Kidney Disease",
        "ckd": "Kidney Disease",
    }
    if d_norm in aliases:
        return aliases[d_norm]

    # Preserve exact known canonical labels.
    known = {"Diabetes", "Heart Disease", "Liver Disease", "Kidney Disease"}
    if d in known:
        return d

    # Best-effort fallback: title-case for display, but may not match a mapping.
    return d


@dataclass(frozen=True)
class GuidanceBundle:
    doctor_to_consult: str
    lifestyle_tips: List[str]
    diet_tips: List[str]
    red_flags: List[str]
    workup_tips: List[str] = field(default_factory=list)
    medication_topics: List[str] = field(default_factory=list)
    diet_template: List[str] = field(default_factory=list)


DOCTOR_BY_DISEASE: Dict[str, str] = {
    "Diabetes": "Primary care clinician or Endocrinologist",
    "Heart Disease": "Primary care clinician or Cardiologist",
    "Kidney Disease": "Primary care clinician or Nephrologist",
    "Liver Disease": "Primary care clinician or Hepatologist/Gastroenterologist",
}


GENERAL_LIFESTYLE_TIPS: List[str] = [
    "Aim for regular physical activity that fits your ability (e.g., brisk walking).",
    "Prioritize 7–9 hours of sleep and consistent sleep times.",
    "If you smoke, consider a quit plan with professional support.",
    "Limit alcohol and avoid binge drinking.",
    "Keep follow-up labs/measurements consistent (BP, glucose/A1c, lipids) if your clinician recommends.",
]


LIFESTYLE_TIPS_BY_DISEASE: Dict[str, List[str]] = {
    "Diabetes": [
        "Discuss an HbA1c monitoring plan and individualized targets with a clinician.",
        "Ask about screening intervals for eyes, kidneys, and feet (typical in diabetes care).",
        "If you use home glucose monitoring, review when/how often with a clinician.",
    ],
    "Heart Disease": [
        "Review blood pressure targets and home BP technique if you track BP.",
        "Discuss cholesterol goals and cardiovascular risk reduction strategies.",
        "If you have chest pain, exertional shortness of breath, or fainting, seek urgent evaluation.",
    ],
    "Kidney Disease": [
        "Discuss how often to check creatinine/eGFR and urine albumin/protein.",
        "Avoid frequent NSAID use (e.g., ibuprofen/naproxen) unless a clinician says it's safe for you.",
        "Review blood pressure control; kidney health is closely linked with BP.",
    ],
    "Liver Disease": [
        "Discuss follow-up liver enzymes and whether imaging (e.g., ultrasound) is appropriate.",
        "Avoid alcohol until you have clarity on the cause of liver risk.",
        "Ask whether viral hepatitis screening is appropriate based on risk factors and labs.",
    ],
}


DIET_TIPS_BY_DISEASE: Dict[str, List[str]] = {
    "Diabetes": [
        "Favor high-fiber carbs (vegetables, legumes, whole grains) and reduce sugary drinks.",
        "Balance meals with protein + fiber to reduce glucose spikes.",
        "Prefer minimally processed foods; watch portion sizes for refined starches and sweets.",
    ],
    "Heart Disease": [
        "Prefer unsaturated fats (olive oil, nuts, fish) and reduce trans/saturated fats.",
        "Moderate sodium intake, especially if blood pressure is elevated.",
        "Increase soluble fiber (e.g., oats, beans) which can help with LDL in some people.",
    ],
    "Kidney Disease": [
        "Discuss sodium, protein, and potassium targets with a clinician/dietitian (needs vary by stage).",
        "Stay well-hydrated unless your clinician has given fluid restrictions.",
        "If potassium/phosphorus restrictions apply, a renal dietitian can tailor food choices safely.",
    ],
    "Liver Disease": [
        "Avoid or minimize alcohol; it can worsen many liver conditions.",
        "Choose balanced meals; avoid frequent high-sugar/high-fat ultra-processed foods.",
        "If weight loss is appropriate, gradual and sustained loss is generally preferred over crash diets.",
    ],
}


WORKUP_TIPS_BY_DISEASE: Dict[str, List[str]] = {
    "Diabetes": [
        "Confirm glycemic status with HbA1c and/or fasting glucose per clinician guidance.",
        "Review kidney screening (urine albumin-to-creatinine ratio) and blood pressure.",
        "Check lipid profile and cardiovascular risk factors periodically.",
    ],
    "Heart Disease": [
        "Review blood pressure and lipid profile; consider an overall ASCVD risk assessment.",
        "Discuss whether an ECG, stress test, or imaging is appropriate based on symptoms and risk.",
        "Assess diabetes status (HbA1c/fasting glucose) as it impacts cardiovascular risk.",
    ],
    "Kidney Disease": [
        "Repeat creatinine/eGFR and urine albumin/protein to confirm chronicity and severity.",
        "Review blood pressure and medications that can affect kidneys.",
        "Consider secondary causes if kidney risk is high (clinician decision).",
    ],
    "Liver Disease": [
        "Repeat liver enzymes and bilirubin; evaluate trends rather than a single value.",
        "Discuss viral hepatitis testing and metabolic risk evaluation (weight, glucose, lipids).",
        "Consider liver ultrasound if clinically indicated.",
    ],
}


MEDICATION_TOPICS_BY_DISEASE: Dict[str, List[str]] = {
    "Diabetes": [
        "Common type 2 diabetes first-line medication is often metformin (depends on kidney function and tolerance).",
        "For cardio-renal benefit, clinicians may consider SGLT2 inhibitors or GLP-1 receptor agonists (patient-specific).",
        "If glucose is very high or symptomatic, insulin may be considered (clinician-guided).",
    ],
    "Heart Disease": [
        "For prevention, clinicians often discuss statins for LDL lowering when overall risk is elevated.",
        "Blood pressure medications (e.g., ACE inhibitors/ARBs, thiazides, calcium-channel blockers) are chosen based on individual factors.",
        "Aspirin/antiplatelet therapy is not appropriate for everyone; it depends on diagnosis and bleeding risk.",
    ],
    "Kidney Disease": [
        "ACE inhibitors or ARBs are commonly used for hypertension and proteinuria protection (clinician-guided).",
        "SGLT2 inhibitors can slow kidney disease progression in some patients (eligibility depends on eGFR/diagnosis).",
        "Medication doses often need adjustment in CKD; a clinician/pharmacist can review safely.",
    ],
    "Liver Disease": [
        "Medication depends strongly on the cause (fatty liver, viral hepatitis, alcohol-related, autoimmune, etc.).",
        "Clinicians may focus on metabolic control (weight, diabetes, lipids) when fatty liver is suspected.",
        "Avoid starting new supplements/herbals without clinician review; some can harm the liver.",
    ],
}


DIET_TEMPLATE_BY_DISEASE: Dict[str, List[str]] = {
    "Diabetes": [
        "Plate method: 1/2 non-starchy vegetables, 1/4 lean protein, 1/4 high-fiber carbs.",
        "Prefer water/unsweetened drinks; limit sugary beverages.",
        "Aim for consistent carb portions; choose low-glycemic, high-fiber options.",
    ],
    "Heart Disease": [
        "Mediterranean/DASH pattern: vegetables, fruits, legumes, whole grains, fish; olive oil as main fat.",
        "Limit sodium and ultra-processed foods; cook more at home when possible.",
        "Reduce saturated fat (processed meats, butter); prefer nuts/seeds and fatty fish.",
    ],
    "Kidney Disease": [
        "Start with sodium reduction; stage-specific potassium/phosphorus advice requires labs.",
        "Protein targets vary by CKD stage; ask for a renal dietitian plan if risk is elevated.",
        "If you have fluid restrictions, follow clinician guidance (not everyone needs this).",
    ],
    "Liver Disease": [
        "Mediterranean-style diet is commonly recommended for metabolic fatty liver patterns.",
        "Avoid alcohol; reduce added sugars (especially sweetened drinks).",
        "If weight loss is appropriate, aim for gradual loss with adequate protein.",
    ],
}


RED_FLAGS_BY_DISEASE: Dict[str, List[str]] = {
    "Diabetes": [
        "Very high blood sugars with dehydration, confusion, or vomiting",
        "Chest pain, severe shortness of breath, or fainting",
    ],
    "Heart Disease": [
        "Chest pain/pressure, pain radiating to arm/jaw",
        "Severe shortness of breath, sudden sweating, or fainting",
    ],
    "Kidney Disease": [
        "Significant swelling of legs/face, very low urine output",
        "Severe shortness of breath or chest pain",
    ],
    "Liver Disease": [
        "Yellowing of skin/eyes (jaundice), confusion, severe abdominal swelling",
        "Vomiting blood or black/tarry stools",
    ],
}


def _normalize_risk_level(level: Optional[str]) -> RiskLevel:
    if not level:
        return "Unknown"
    level = level.strip().lower()
    if level.startswith("low"):
        return "Low"
    if level.startswith("mod"):
        return "Moderate"
    if level.startswith("med"):
        return "Moderate"
    if level.startswith("high"):
        return "High"
    return "Unknown"


def get_overall_guidance(disease_levels: Dict[str, str]) -> GuidanceBundle:
    """Merge per-disease guidance into one bundle, prioritizing elevated-risk diseases.

    Args:
        disease_levels: {disease_name: risk_level} e.g. {"Diabetes": "HIGH", ...}
    """
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ordered = sorted(disease_levels.items(), key=lambda kv: rank.get(str(kv[1]).upper(), 3))
    elevated = [d for d, lvl in ordered if str(lvl).upper() in ("HIGH", "MEDIUM")]
    focus = elevated if elevated else [d for d, _ in ordered[:2]]

    doctors: List[str] = []
    merged: Dict[str, List[str]] = {
        "lifestyle": [], "diet": [], "red_flags": [],
        "workup": [], "meds": [], "diet_template": [],
    }
    for d in focus:
        b = get_guidance_for_disease(d, risk_level=disease_levels[d])
        if b.doctor_to_consult not in doctors:
            doctors.append(b.doctor_to_consult)
        for key, src in [
            ("lifestyle", b.lifestyle_tips), ("diet", b.diet_tips),
            ("red_flags", b.red_flags), ("workup", b.workup_tips),
            ("meds", b.medication_topics), ("diet_template", b.diet_template),
        ]:
            for item in src:
                if item not in merged[key]:
                    merged[key].append(item)

    return GuidanceBundle(
        doctor_to_consult="; ".join(doctors) if doctors else "Primary care clinician",
        lifestyle_tips=merged["lifestyle"],
        diet_tips=merged["diet"],
        red_flags=merged["red_flags"],
        workup_tips=merged["workup"],
        medication_topics=merged["meds"],
        diet_template=merged["diet_template"],
    )


def get_guidance_for_disease(disease: str, risk_level: Optional[str] = None) -> GuidanceBundle:
    """Return conservative, rule-based guidance."""

    disease_key = _normalize_disease_name(disease)

    doctor = DOCTOR_BY_DISEASE.get(disease_key, "Primary care clinician")

    disease_diet = DIET_TIPS_BY_DISEASE.get(disease_key, [])
    red_flags = RED_FLAGS_BY_DISEASE.get(disease_key, [
        "Severe chest pain, trouble breathing, fainting, confusion, or uncontrolled bleeding",
    ])

    # Optional: tailor the tone based on risk, without changing content.
    level_norm = _normalize_risk_level(risk_level)
    lifestyle = []
    diet = list(disease_diet)

    if level_norm == "High":
        lifestyle.append("Consider booking a clinician visit soon to review risk factors and next steps.")
    elif level_norm == "Moderate":
        lifestyle.append("Consider a routine clinician follow-up to discuss prevention and monitoring.")
    elif level_norm == "Low":
        lifestyle.append("Keep up healthy habits and routine preventive checkups.")

    lifestyle.extend(LIFESTYLE_TIPS_BY_DISEASE.get(disease_key, []))
    lifestyle.extend(GENERAL_LIFESTYLE_TIPS)

    workup = list(WORKUP_TIPS_BY_DISEASE.get(disease_key, []))
    meds = list(MEDICATION_TOPICS_BY_DISEASE.get(disease_key, []))
    diet_template = list(DIET_TEMPLATE_BY_DISEASE.get(disease_key, []))

    return GuidanceBundle(
        doctor_to_consult=doctor,
        lifestyle_tips=lifestyle,
        diet_tips=diet,
        red_flags=red_flags,
        workup_tips=workup,
        medication_topics=meds,
        diet_template=diet_template,
    )
