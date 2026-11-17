"""Patient PDF report generation for the Multi-Disease Risk Prediction system.

Generates a single, clinician-readable PDF covering ALL FOUR diseases:
- Predicted risk scores + risk levels
- SHAP-based reasoning (top factors increasing / decreasing risk)
- Cautions & red flags (when to seek care)
- Doctors to consult + follow-up workup, diet and lifestyle guidance
- Patient-reported lifestyle/history (from the NLP text extractor, if provided)
- Missing-feature disclosure + methodology note + disclaimer
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fpdf import FPDF

from src.guidance.rules import get_guidance_for_disease
from src.inference.engine import (
    DISEASE_NAMES,
    DISEASE_PANELS,
    DISCLAIMER,
    FEATURE_LABELS,
    NORMAL_RANGES,
    Artifacts,
    disease_readiness,
    format_raw_value,
    provided_and_missing,
    risk_level,
    structured_factors,
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────

C_DARK    = (15, 23, 42)     # slate-900
C_ACCENT  = (3, 105, 161)    # sky-700
C_MUTED   = (100, 116, 139)  # slate-500
C_LOW     = (34, 197, 94)    # green-500
C_MED     = (245, 158, 11)   # amber-500
C_HIGH    = (239, 68, 68)    # red-500
C_ROW_ALT = (241, 245, 249)  # slate-100
C_BORDER  = (203, 213, 225)  # slate-300

LEVEL_COLORS = {"LOW": C_LOW, "MEDIUM": C_MED, "HIGH": C_HIGH}

PAGE_W = 210
MARGIN = 14
CONTENT_W = PAGE_W - 2 * MARGIN


def _latin(text: Any) -> str:
    """Make text safe for core PDF fonts (latin-1)."""
    s = str(text)
    s = (s.replace("—", "-").replace("–", "-").replace("→", "->")
           .replace("≈", "~").replace("≥", ">=").replace("≤", "<=")
           .replace("×", "x").replace("Δ", "Delta").replace("²", "2"))
    return s.encode("latin-1", "replace").decode("latin-1")


def _risk_bar_color(level: str):
    return LEVEL_COLORS.get(level, C_MUTED)


def _disp(factor_item: Dict[str, Any]) -> str:
    """Display value for a SHAP factor, with special handling for gender."""
    if factor_item.get("feature") == "gender":
        return format_raw_value("gender", factor_item.get("raw_value"))
    return str(factor_item.get("raw_value_display", ""))


# ─────────────────────────────────────────────────────────────────────────────
# PDF DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

class PatientReport(FPDF):
    def __init__(self, patient_name: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.patient_name = patient_name or "Patient"
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(MARGIN, 14, MARGIN)

    def header(self):
        self.set_fill_color(*C_DARK)
        self.rect(0, 0, PAGE_W, 22, "F")
        self.set_y(5)
        self.set_x(MARGIN)
        self.set_font("helvetica", "B", 15)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "MedRisk AI - Multi-Disease Risk Assessment Report", new_x="LMARGIN", new_y="NEXT")
        self.set_x(MARGIN)
        self.set_font("helvetica", "", 8.5)
        self.set_text_color(125, 211, 252)
        self.cell(0, 5, "XGBoost risk models - SHAP explanations - IIIT Sri City BTP 2025-26", new_x="LMARGIN", new_y="NEXT")
        self.set_y(28)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*C_BORDER)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.set_y(-11)
        self.set_font("helvetica", "I", 7)
        self.set_text_color(*C_MUTED)
        self.multi_cell(0, 3.5, _latin(f"For research/education only - not a medical diagnosis. {DISCLAIMER}"), align="C")

    # ── building blocks ─────────────────────────────────────────────────────

    def section_title(self, title: str):
        self.ln(2)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(*C_ACCENT)
        self.cell(0, 7, _latin(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.4)
        self.line(MARGIN, self.get_y(), MARGIN + 40, self.get_y())
        self.ln(1.5)
        self.set_line_width(0.2)

    def kv_row(self, label: str, value: str, alt: bool = False):
        if alt:
            self.set_fill_color(*C_ROW_ALT)
        self.set_font("helvetica", "", 9)
        self.set_text_color(*C_MUTED)
        self.cell(70, 6, _latin(label), border=0, fill=alt)
        self.set_text_color(*C_DARK)
        self.set_font("helvetica", "B", 9)
        self.cell(0, 6, _latin(value), border=0, fill=alt, new_x="LMARGIN", new_y="NEXT")

    def body(self, text: str, size: float = 9, style: str = "", color=C_DARK):
        self.set_font("helvetica", style, size)
        self.set_text_color(*color)
        self.multi_cell(0, 4.6, _latin(text))

    def bullet(self, text: str, color=C_DARK, indent: float = 4):
        self.set_text_color(*color)
        self.set_font("helvetica", "", 9)
        x = MARGIN + indent
        self.set_x(x)
        self.cell(3, 4.6, chr(149))
        self.set_x(x + 4)
        self.multi_cell(CONTENT_W - indent - 4, 4.6, _latin(text), new_x="LMARGIN", new_y="NEXT")

    def risk_gauge(self, prob: float, level: str, width: float = CONTENT_W):
        """Horizontal probability bar with LOW/MED/HIGH zones."""
        x0, y0 = self.get_x(), self.get_y()
        h = 6
        self.set_fill_color(226, 232, 240)
        self.rect(x0, y0, width, h, "F")
        color = _risk_bar_color(level)
        fill_w = max(1.0, width * min(max(prob, 0.0), 1.0))
        self.set_fill_color(*color)
        self.rect(x0, y0, fill_w, h, "F")
        # zone ticks at 35% and 65%
        self.set_draw_color(148, 163, 184)
        for t in (0.35, 0.65):
            self.line(x0 + width * t, y0 + h, x0 + width * t, y0 + h + 1.2)
        self.set_y(y0 + h + 2)
        self.set_font("helvetica", "", 7)
        self.set_text_color(*C_MUTED)
        self.cell(width * 0.35, 3, "LOW (<35%)", align="C")
        self.cell(width * 0.30, 3, "MEDIUM (35-65%)", align="C")
        self.cell(width * 0.35, 3, "HIGH (>65%)", align="C", new_x="LMARGIN", new_y="NEXT")


def _shap_chart_png(shap_row: np.ndarray, numeric_cols: List[str], patient_dict: Dict[str, Any],
                    disease_name: str, out_path: Path, top_n: int = 10) -> Optional[Path]:
    """Render a diverging horizontal bar chart of SHAP contributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    shap_row = np.asarray(shap_row, dtype=np.float32)
    order = np.argsort(np.abs(shap_row))[::-1][:top_n]
    order = order[np.argsort(shap_row[order])]  # sort by value for diverging look

    labels, vals, colors = [], [], []
    for i in order:
        feat = numeric_cols[int(i)]
        raw = patient_dict.get(feat)
        val_str = format_raw_value(feat, raw)
        labels.append(f"{FEATURE_LABELS.get(feat, feat)}  [{val_str}]")
        v = float(shap_row[int(i)])
        vals.append(v)
        colors.append("#ef4444" if v > 0 else "#22c55e")

    fig_h = max(2.2, 0.32 * len(labels) + 0.9)
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=150)
    ax.barh(range(len(vals)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="#64748b", lw=0.8)
    ax.set_xlabel("SHAP contribution (log-odds)  |  red = raises risk, green = lowers risk", fontsize=7.5)
    ax.set_title(f"{disease_name}: top feature drivers", fontsize=9, color="#0f172a")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_patient_report(
    *,
    patient_dict: Dict[str, Any],
    results: Dict[str, float],
    artifacts: Artifacts,
    patient_name: str = "Patient",
    history_text: Optional[str] = None,
    lifestyle_features: Optional[Any] = None,
    top_k: int = 8,
) -> bytes:
    """Build the full multi-disease PDF report. Returns PDF bytes."""

    pdf = PatientReport(patient_name)
    pdf.add_page()
    numeric_cols = list(artifacts.numeric_cols)
    provided, missing = provided_and_missing(patient_dict, numeric_cols)
    readiness = disease_readiness(patient_dict)

    # ── 1. Patient + inputs summary ─────────────────────────────────────────
    pdf.section_title("Patient Summary")
    pdf.kv_row("Name", patient_name or "Patient", alt=True)
    pdf.kv_row("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M"))
    pdf.kv_row("Model", "XGBoost (per-disease binary risk models)", alt=True)
    if artifacts.mean_auc is not None:
        pdf.kv_row("Validation mean AUC (4 diseases)", f"{artifacts.mean_auc:.4f}")
    pdf.kv_row("Lab values provided", f"{len(provided)} of {len(numeric_cols)} features", alt=True)
    if missing:
        pdf.kv_row("Auto-handled missing", f"{len(missing)} features (sentinel encoding)")

    if provided:
        pdf.ln(1)
        pdf.body("Provided values: " + ", ".join(
            f"{FEATURE_LABELS.get(f, f)} = {format_raw_value(f, patient_dict.get(f))}" for f in provided
        ), size=8, color=C_MUTED)

    # ── 2. Patient-reported lifestyle & history (NLP) ───────────────────────
    if history_text or lifestyle_features is not None:
        pdf.section_title("Patient-Reported Lifestyle & History (from text)")
        if history_text:
            pdf.body(f'Patient text: "{history_text}"', size=8, color=C_MUTED)
            pdf.ln(1)
        if lifestyle_features is not None:
            lf = lifestyle_features
            rows = [
                ("Smoking", f"{lf.smoking_status}" + (f" ({lf.smoking_frequency})" if lf.smoking_frequency else "")),
                ("Alcohol", f"{lf.alcohol_status}" + (f" ({lf.alcohol_frequency})" if lf.alcohol_frequency else "")),
                ("Physical activity", f"{lf.physical_activity}"),
            ]
            if lf.family_history:
                rows.append(("Family history", ", ".join(lf.family_history)))
            if lf.symptoms:
                rows.append(("Reported symptoms", ", ".join(lf.symptoms)))
            for i, (k, v) in enumerate(rows):
                if v and v != "unknown":
                    pdf.kv_row(k, v, alt=(i % 2 == 0))

    # ── 3. Overall risk summary ─────────────────────────────────────────────
    pdf.section_title("Overall Risk Summary")
    col_w = [70, 40, 35, 37]
    pdf.set_fill_color(*C_DARK)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 9)
    for h, w in zip(["Disease", "Risk Score", "Risk Level", "Interpretation"], col_w):
        pdf.cell(w, 7, h, border=0, fill=True, align="C")
    pdf.ln()

    interp = {"LOW": "Within lower-risk range", "MEDIUM": "Moderate - monitor", "HIGH": "Elevated - act on factors"}
    for i, (target, name) in enumerate(zip(_targets(), DISEASE_NAMES)):
        alt = i % 2 == 0
        scoreable = readiness[target]["scoreable"]
        prob = float(results[target])
        level = risk_level(prob)
        if alt:
            pdf.set_fill_color(*C_ROW_ALT)
        pdf.set_text_color(*C_DARK)
        pdf.set_font("helvetica", "", 9)
        pdf.cell(col_w[0], 7, _latin(name), border=0, fill=alt, align="C")
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(col_w[1], 7, f"{prob * 100:.1f}%", border=0, fill=alt, align="C")
        pdf.set_text_color(*_risk_bar_color(level))
        pdf.cell(col_w[2], 7, level + ("" if scoreable else "*"), border=0, fill=alt, align="C")
        pdf.set_text_color(*C_MUTED)
        pdf.set_font("helvetica", "", 8)
        note = interp[level] if scoreable else "Few markers - interpret with caution"
        pdf.cell(col_w[3], 7, _latin(note), border=0, fill=alt, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 7.5)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 4, _latin("* Cross-disease estimate only - no disease-specific markers provided; "
                          "enter the markers listed on the disease page for better accuracy."),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ── 4. Per-disease detail sections ──────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="btp_report_"))
    for target, name in zip(_targets(), DISEASE_NAMES):
        prob = float(results[target])
        level = risk_level(prob)
        scoreable = readiness[target]["scoreable"]
        guidance = get_guidance_for_disease(disease=name, risk_level=level if scoreable else "Unknown")

        pdf.add_page()
        pdf.section_title(f"{name} - Detailed Assessment")

        factors = structured_factors(target, patient_dict, artifacts, top_k=top_k)

        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(*_risk_bar_color(level))
        pdf.cell(30, 10, f"{prob * 100:.1f}%")
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 10, f"{level} RISK", new_x="LMARGIN", new_y="NEXT")
        pdf.risk_gauge(prob, level)
        pdf.ln(1)

        if not scoreable:
            panel_full = ", ".join(FEATURE_LABELS.get(f, f)
                                   for f in DISEASE_PANELS.get(target, []))
            pdf.set_fill_color(254, 243, 199)  # amber-100
            pdf.set_text_color(146, 64, 14)    # amber-800
            pdf.set_font("helvetica", "", 8.5)
            pdf.multi_cell(0, 4.5, _latin(
                f"Confidence note: this score is a cross-disease estimate based on the other "
                f"features provided. No {name.lower()}-specific markers were entered — for a "
                f"more accurate assessment, provide at least one of: {panel_full}."
            ), border=0, fill=True)
            pdf.ln(1)

        # reasoning
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*C_DARK)
        pdf.cell(0, 6, "Why this result (top contributing factors):", new_x="LMARGIN", new_y="NEXT")

        inc = factors.get("top_increasing", [])
        dec = factors.get("top_decreasing", [])
        if inc:
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_text_color(*C_HIGH)
            pdf.cell(0, 5, "Factors increasing risk:", new_x="LMARGIN", new_y="NEXT")
            for it in inc[:4]:
                status = f" [{it['status']}]" if it.get("status") not in (None, "unknown", "missing") else ""
                pdf.bullet(f"{it['label']}: {_disp(it)}{status} ({it['impact_strength']} impact)", color=C_DARK)
        if dec:
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_text_color(*C_LOW)
            pdf.cell(0, 5, "Factors decreasing risk:", new_x="LMARGIN", new_y="NEXT")
            for it in dec[:3]:
                status = f" [{it['status']}]" if it.get("status") not in (None, "unknown", "missing") else ""
                pdf.bullet(f"{it['label']}: {_disp(it)}{status} ({it['impact_strength']} impact)", color=C_DARK)
        if not inc and not dec:
            pdf.body("No dominant feature drivers detected; result driven by overall profile.", size=8.5, color=C_MUTED)

        # SHAP chart
        png = _shap_chart_png(factors["shap_row"], numeric_cols, patient_dict, name,
                              tmp_dir / f"shap_{target}.png", top_n=7)
        if png and png.exists():
            chart_w = CONTENT_W * 0.86
            pdf.image(str(png), x=MARGIN + (CONTENT_W - chart_w) / 2, w=chart_w)
            pdf.ln(0.5)

        # guidance
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*C_DARK)
        pdf.cell(0, 6, "What to do next:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_text_color(*C_ACCENT)
        pdf.cell(0, 5, f"Consult: {guidance.doctor_to_consult}", new_x="LMARGIN", new_y="NEXT")
        if guidance.workup_tips:
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_text_color(*C_DARK)
            pdf.cell(0, 5, "Follow-up / workup:", new_x="LMARGIN", new_y="NEXT")
            for t in guidance.workup_tips[:3]:
                pdf.bullet(t)
        if guidance.diet_template:
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_text_color(*C_DARK)
            pdf.cell(0, 5, "Diet template:", new_x="LMARGIN", new_y="NEXT")
            for t in guidance.diet_template[:3]:
                pdf.bullet(t)
        if guidance.lifestyle_tips:
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_text_color(*C_DARK)
            pdf.cell(0, 5, "Lifestyle:", new_x="LMARGIN", new_y="NEXT")
            for t in guidance.lifestyle_tips[:2]:
                pdf.bullet(t)

        # red flags
        if guidance.red_flags:
            pdf.ln(1)
            pdf.set_fill_color(254, 226, 226)
            y0 = pdf.get_y()
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_text_color(*C_HIGH)
            pdf.cell(0, 5, "Seek medical care promptly if you experience:", new_x="LMARGIN", new_y="NEXT")
            for t in guidance.red_flags:
                pdf.bullet(t, color=C_DARK)

    # ── 5. Methodology + disclosure ─────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Methodology & Disclosures")
    pdf.body(
        "Risk scores are produced by per-disease XGBoost classifiers trained on a unified "
        "24-feature clinical schema combining four public datasets (diabetes, heart, liver, "
        "kidney cohorts; ~191k samples after de-duplication). Features not measured for a "
        "patient are encoded with a sentinel value (-999) so the model can still score "
        "partial lab panels. Factor contributions are TreeSHAP values (XGBoost "
        "pred_contribs) expressed in log-odds; positive values raise the predicted risk."
    )
    pdf.ln(1)
    if missing:
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*C_DARK)
        pdf.cell(0, 5, "Features auto-handled (not provided):", new_x="LMARGIN", new_y="NEXT")
        pdf.body(", ".join(FEATURE_LABELS.get(f, f) for f in missing), size=8, color=C_MUTED)
        pdf.ln(1)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(*C_DARK)
    pdf.cell(0, 5, "Important notes:", new_x="LMARGIN", new_y="NEXT")
    for t in [
        "Scores are statistical risk estimates, not diagnoses; a HIGH result means 'discuss with a clinician', not 'you have the disease'.",
        "Disease risk levels use fixed bands (LOW <35%, MEDIUM 35-65%, HIGH >65%).",
        "Models were trained on public research datasets and may not generalise to all populations.",
        "Cross-disease comorbidity notes in the app are clinical heuristics, not model outputs.",
    ]:
        pdf.bullet(t)

    return bytes(pdf.output())


def _targets() -> List[str]:
    from src.inference.engine import TARGET_COLS
    return list(TARGET_COLS)
