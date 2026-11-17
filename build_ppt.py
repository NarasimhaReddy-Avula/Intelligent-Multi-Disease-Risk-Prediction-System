#!/usr/bin/env python3
"""Build BTP final defense deck — dark navy/teal, 15 slides, real figures."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
import os

FIG = "report_figures"
OUT = "BTP_Final_Presentation.pptx"

# ---- palette (matches Streamlit app) ----
BG      = RGBColor(0x0B, 0x12, 0x20)   # deep navy
CARD    = RGBColor(0x15, 0x20, 0x34)   # slightly lighter panel
INK     = RGBColor(0xF8, 0xFA, 0xFC)   # near-white
MUTED   = RGBColor(0x94, 0xA3, 0xB8)   # slate
TEAL    = RGBColor(0x2D, 0xD4, 0xBF)   # accent
BLUE    = RGBColor(0x60, 0xA5, 0xFA)
AMBER   = RGBColor(0xFB, 0xBF, 0x24)
RED     = RGBColor(0xF8, 0x71, 0x71)
GREEN   = RGBColor(0x4A, 0xDE, 0x80)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)

SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    r.fill.solid(); r.fill.fore_color.rgb = BG
    r.line.fill.background(); r.shadow.inherit = False
    return s


def box(s, l, t, w, h, color=CARD, line=None, round_=True):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE,
        l, t, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    if line: shp.line.color.rgb = line; shp.line.width = Pt(1)
    else: shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def text(s, l, t, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=6, line_spacing=1.0):
    """runs: list of paragraphs; each = list of (txt, size, color, bold) or dict"""
    tb = s.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        if isinstance(para, dict):
            p.alignment = para.get("align", align)
            p.space_after = Pt(para.get("space_after", space_after))
            p.line_spacing = para.get("line_spacing", line_spacing)
            runs_ = para["runs"]
        else:
            runs_ = para
        for txt, size, color, bold in runs_:
            r = p.add_run(); r.text = txt
            r.font.size = Pt(size); r.font.color.rgb = color
            r.font.bold = bold; r.font.name = "Segoe UI"
    return tb


def header(s, title, kicker=None):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(0.42),
                             Inches(0.09), Inches(0.62))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    bar.shadow.inherit = False
    paras = []
    if kicker:
        paras.append([(kicker, 12, TEAL, True)])
    paras.append([(title, 28, INK, True)])
    text(s, Inches(0.8), Inches(0.34) if kicker else Inches(0.44),
         Inches(12), Inches(1.0), paras, space_after=0)


def bullets(s, l, t, w, h, items, size=16, gap=10, dot_color=TEAL):
    paras = []
    for it in items:
        if isinstance(it, tuple):   # (bold_head, rest)
            head, rest = it
            paras.append([("▸  ", size, dot_color, True),
                          (head, size, INK, True),
                          (rest, size, MUTED, False)])
        else:
            paras.append([("▸  ", size, dot_color, True),
                          (it, size, RGBColor(0xE2, 0xE8, 0xF0), False)])
    text(s, l, t, w, h, paras, space_after=gap, line_spacing=1.05)


def figure(s, path, l, t, w, h, pad=0.12):
    """White card + image fitted inside box (l,t,w,h inches)."""
    L, T, W, H = Inches(l), Inches(t), Inches(w), Inches(h)
    box(s, L, T, W, H, color=WHITE)
    iw, ih = Image.open(path).size
    avail_w, avail_h = w - 2 * pad, h - 2 * pad
    scale = min(avail_w / (iw / 96), avail_h / (ih / 96))
    pw, ph = iw / 96 * scale, ih / 96 * scale
    pl, pt = l + (w - pw) / 2, t + (h - ph) / 2
    s.shapes.add_picture(path, Inches(pl), Inches(pt),
                         Inches(pw), Inches(ph))


def footer(s, n):
    text(s, Inches(0.55), Inches(7.08), Inches(6), Inches(0.35),
         [[("Multi-Disease Risk Prediction · BTP Final", 10, MUTED, False)]],
         space_after=0)
    text(s, Inches(12.3), Inches(7.08), Inches(0.7), Inches(0.35),
         [[(str(n), 11, TEAL, True)]], align=PP_ALIGN.RIGHT, space_after=0)


# ══════════════════ SLIDE 1 — TITLE ══════════════════
s = slide()
box(s, Inches(0.9), Inches(1.15), Inches(11.53), Inches(5.2), color=CARD)
text(s, Inches(1.4), Inches(1.55), Inches(10.5), Inches(0.5),
     [[("BACHELOR THESIS PROJECT · FINAL PRESENTATION", 13, TEAL, True)]],
     align=PP_ALIGN.CENTER)
text(s, Inches(1.4), Inches(2.15), Inches(10.5), Inches(1.8),
     [[("Intelligent Multi-Disease Risk Prediction System", 36, INK, True)],
      [("with Explainability", 36, INK, True)]],
     align=PP_ALIGN.CENTER, space_after=4)
text(s, Inches(1.4), Inches(3.95), Inches(10.5), Inches(1.4),
     [[("Diabetes  ·  Heart Disease  ·  Liver Disease  ·  Kidney Disease",
        16, TEAL, True)],
      [(" ", 8, MUTED, False)],
      [("Avula Lakshmi Narasimha Reddy (S20230010030)   ·   "
        "Rayapu Nishanth (S20230010202)", 15, INK, False)],
      [("Guide: Dr. R. Selvi   ·   IIIT Sri City   ·   2025–2026",
        13, MUTED, False)]],
     align=PP_ALIGN.CENTER, space_after=6)

# ══════════════════ SLIDE 2 — PROBLEM ══════════════════
s = slide(); header(s, "The Problem", "MOTIVATION")
bullets(s, Inches(0.8), Inches(1.55), Inches(7.2), Inches(5), [
    ("Multi-morbidity is the norm. ",
     "Chronic diseases co-occur — diabetes drives kidney and heart disease; "
     "yet most public datasets and tools treat each condition alone."),
    ("Data is fragmented. ",
     "Each disease cohort measures a different biomarker panel — "
     "no shared feature space, so models can't be combined naïvely."),
    ("Imputation corrupts. ",
     "Standard KNN/median imputation invents labs that were never "
     "measured, fabricating medically implausible correlations."),
    ("Black boxes don't fly in clinics. ",
     "A risk score without 'why' is unusable — explanations must be "
     "per-patient and patient-readable."),
], size=17, gap=14)
box(s, Inches(8.35), Inches(1.55), Inches(4.35), Inches(4.9))
text(s, Inches(8.7), Inches(1.9), Inches(3.7), Inches(4.4),
     [[("THE GAP", 12, TEAL, True)],
      [("No unified schema across cohorts", 15, INK, False)],
      [("No explicit missingness handling", 15, INK, False)],
      [("No proof one disease's data helps another", 15, INK, False)],
      [("No single report covering all diseases", 15, INK, False)],
      [(" ", 6, MUTED, False)],
      [("This project closes all four gaps.", 15, TEAL, True)]],
     space_after=12)
footer(s, 2)

# ══════════════════ SLIDE 3 — QUESTION & OBJECTIVES ══════════════════
s = slide(); header(s, "Research Questions", "WHAT WE SET OUT TO ANSWER")
box(s, Inches(0.8), Inches(1.5), Inches(5.9), Inches(2.35))
text(s, Inches(1.15), Inches(1.75), Inches(5.2), Inches(1.9),
     [[("RQ1 — FUSION", 12, TEAL, True)],
      [("Can heterogeneous disease cohorts be fused into one schema "
        "without corrupting the signal?", 17, INK, False)]], space_after=8)
box(s, Inches(0.8), Inches(4.05), Inches(5.9), Inches(2.35))
text(s, Inches(1.15), Inches(4.3), Inches(5.2), Inches(1.9),
     [[("RQ2 — TRANSFER", 12, TEAL, True)],
      [("Does data measured for one disease carry predictive signal "
        "for another?", 17, INK, False)]], space_after=8)
bullets(s, Inches(7.15), Inches(1.55), Inches(5.55), Inches(5), [
    ("Unify ", "4 cohorts → 24-feature clinical schema (~191K rows)"),
    ("Handle missingness ", "explicitly — masked & sentinel encodings"),
    ("Benchmark ", "LR / RF / XGBoost / LightGBM / FT-Transformer"),
    ("Deploy ", "app + API + single patient PDF over one engine"),
    ("Explain ", "per-patient SHAP, NLP context, LLM guidance"),
    ("Prove ", "cross-disease signal — attribution, ablation, stress test"),
], size=15, gap=9)
footer(s, 3)

# ══════════════════ SLIDE 4 — DATASETS ══════════════════
s = slide(); header(s, "Datasets & Fusion Scope", "SEMESTER 1 · DATA")
rows = [("Cohort", "Samples", "Share", "Label"),
        ("Diabetes", "100,000", "52.3%", "diabetic / non-diabetic"),
        ("Heart", "70,000", "36.6%", "disease / no disease"),
        ("Liver", "30,691", "10.1%", "liver disease / none"),
        ("Kidney", "1,659 → 2,032 (SMOTE)", "1.1%", "CKD / non-CKD"),
        ("Unified (deduplicated)", "≈191,363", "100%", "4 target columns")]
tbl = s.shapes.add_table(6, 4, Inches(0.8), Inches(1.6),
                        Inches(7.3), Inches(3.6)).table
tbl.columns[0].width = Inches(2.9); tbl.columns[1].width = Inches(1.9)
tbl.columns[2].width = Inches(1.1); tbl.columns[3].width = Inches(1.4)
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = val
        p = c.text_frame.paragraphs[0]
        p.font.size = Pt(14 if i else 13)
        p.font.name = "Segoe UI"
        p.font.bold = (i == 0 or i == 5)
        p.font.color.rgb = TEAL if i == 0 else (INK if i == 5 else
                                              RGBColor(0xE2, 0xE8, 0xF0))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD if i % 2 else RGBColor(0x0F, 0x18, 0x2A)
        c.margin_left = Inches(0.08)
box(s, Inches(8.5), Inches(1.6), Inches(4.2), Inches(4.7))
text(s, Inches(8.85), Inches(1.9), Inches(3.55), Inches(4.2),
     [[("KEY DECISION", 12, AMBER, True)],
      [("Breast cancer → Kidney", 15, INK, True)],
      [("Imaging morphology shares no schema with lab markers — it "
        "cannot fuse. Kidney (the proposal's own extension option) "
        "overlaps clinically and enables the cardiorenal story.",
        13, MUTED, False)],
      [(" ", 6, MUTED, False)],
      [("ONE LABEL PER ROW", 12, AMBER, True)],
      [("Each row belongs to one cohort → only its own disease label is "
        "observed. The other three are unobserved, not negative.",
        13, MUTED, False)]], space_after=10)
footer(s, 4)

# ══════════════════ SLIDE 5 — SCHEMA ══════════════════
s = slide(); header(s, "Unified Clinical Schema — 24 Features",
                  "SEMESTER 1 · DATA ENGINEERING")
figure(s, f"{FIG}/unified_schema_24_features.png", 0.55, 1.5, 8.3, 5.35)
bullets(s, Inches(9.1), Inches(1.7), Inches(3.7), Inches(5), [
    ("5 panels: ", "demographics, cardiovascular, diabetes, liver, kidney"),
    ("Aligned units ", "across cohorts; categorical buckets → numeric midpoints"),
    ("BMI derived ", "from height/weight where absent"),
    ("Disease-specific features kept ", "— they become each model's own panel"),
], size=14, gap=10)
footer(s, 5)

# ══════════════════ SLIDE 6 — MISSINGNESS ══════════════════
s = slide(); header(s, "Structural Missingness — Two Encodings",
                  "SEMESTER 1 · CORE DESIGN")
bullets(s, Inches(0.8), Inches(1.55), Inches(6.0), Inches(4.2), [
    ("Structural ≠ random. ", "A feature is absent for an entire cohort "
     "because the panel never measured it."),
    ("Masked encoding ", "— value + binary availability flag, for the "
     "transformer: x̃ⱼ = [xⱼ, mⱼ]"),
    ("Sentinel encoding ", "— missing numerics = −999, far outside the "
     "scaled range; trees isolate it via split rules"),
    ("Hypothesis: ", "sentinel suffices for trees; explicit masks feed "
     "attention models. Tested in Week 6."),
], size=15, gap=12)
figure(s, f"{FIG}/data_pipeline_masked_sentinel.png", 7.0, 1.5, 5.8, 4.3)
box(s, Inches(0.8), Inches(5.95), Inches(12.0), Inches(0.85))
text(s, Inches(1.15), Inches(6.12), Inches(11.3), Inches(0.5),
     [[("Missingness is information — encoded explicitly, never silently "
        "imputed away.", 15, TEAL, True)]], space_after=0)
footer(s, 6)

# ══════════════════ SLIDE 7 — MODELS COMPARED ══════════════════
s = slide(); header(s, "Models Benchmarked", "SEMESTER 1 · BASELINES")
rows = [("Model", "Role", "Heart", "Diab.", "Liver", "Kidney"),
        ("Logistic Reg.", "linear, interpretable", "0.786", "0.933",
         "0.513*", "0.849"),
        ("Random Forest", "bagged ensemble", "0.793", "0.941", "0.999",
         "0.929"),
        ("XGBoost", "boosted trees", "0.791", "0.941", "0.999", "0.936"),
        ("LightGBM", "histogram boosting", "0.794", "0.941", "0.999",
         "0.932")]
tbl = s.shapes.add_table(5, 6, Inches(0.8), Inches(1.65),
                        Inches(7.5), Inches(3.3)).table
tbl.columns[0].width = Inches(1.9); tbl.columns[1].width = Inches(2.0)
for j in range(2, 6): tbl.columns[j].width = Inches(0.9)
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = val
        p = c.text_frame.paragraphs[0]
        p.font.size = Pt(13); p.font.name = "Segoe UI"
        p.font.bold = (i == 0)
        p.font.color.rgb = TEAL if i == 0 else RGBColor(0xE2, 0xE8, 0xF0)
        c.fill.solid()
        c.fill.fore_color.rgb = CARD if i % 2 else RGBColor(0x0F, 0x18, 0x2A)
        c.margin_left = Inches(0.08)
text(s, Inches(0.8), Inches(5.05), Inches(7.4), Inches(1.6),
     [[("Test AUC-ROC, sentinel encoding · 70/15/15 stratified split · "
        "*LR liver collapse explained next slide", 12, MUTED, False)],
      [("Each head trained only on rows where that label is observed — "
        "missing labels are never treated as negatives.", 13, INK, False)]],
     space_after=8)
box(s, Inches(8.7), Inches(1.65), Inches(4.0), Inches(4.9))
text(s, Inches(9.05), Inches(1.95), Inches(3.35), Inches(4.4),
     [[("WHY THESE FOUR?", 12, TEAL, True)],
      [("LR — can a linear model suffice? (clinical interpretability)",
        13, MUTED, False)],
      [("RF — bagged trees, noise-robust", 13, MUTED, False)],
      [("XGB — regularized boosting, tabular SOTA", 13, MUTED, False)],
      [("LGBM — fast histogram variant", 13, MUTED, False)],
      [("+ FT-Transformer — deep multi-output comparison (next)",
        13, INK, False)]], space_after=12)
footer(s, 7)

# ══════════════════ SLIDE 8 — ENCODING FINDING ══════════════════
s = slide(); header(s, "Sentinel vs. Masked — Hypothesis Tested",
                  "SEMESTER 1 · KEY FINDING")
box(s, Inches(0.8), Inches(1.6), Inches(5.7), Inches(2.6))
text(s, Inches(1.15), Inches(1.9), Inches(5.0), Inches(2.1),
     [[("TREE MODELS: ENCODING-AGNOSTIC", 12, TEAL, True)],
      [("Mean |ΔAUC| across 12 tree comparisons = ", 15, INK, False),
       ("0.0013", 20, TEAL, True)],
      [("Trees isolate the −999 sentinel via split rules as easily as "
        "they read a mask column.", 13, MUTED, False)]], space_after=10)
box(s, Inches(0.8), Inches(4.45), Inches(5.7), Inches(2.3))
text(s, Inches(1.15), Inches(4.75), Inches(5.0), Inches(1.85),
     [[("LOGISTIC REGRESSION: LIVER COLLAPSE", 12, RED, True)],
      [("0.513 sentinel  →  0.748 masked ", 20, RED, True),
       ("(+0.235)", 15, AMBER, True)],
      [("The extreme sentinel outlier warps the linear boundary — "
        "sentinel is a tree-only representation.", 13, MUTED, False)]],
     space_after=10)
bullets(s, Inches(7.0), Inches(1.7), Inches(5.7), Inches(5), [
    ("Verified the Week-4 hypothesis ", "before committing to deployment"),
    ("Disease-level insight: ", "liver is non-linearly separable "
     "(tree AUC >0.999 vs LR ≈0.5)"),
    ("Heart is the hardest task ", "— only 29% schema coverage; "
     "no troponin/ECG features"),
    ("Kidney: small cohort ", "(2K rows) shows slight masked advantage "
     "— masks help when data is scarce"),
], size=15, gap=13)
footer(s, 8)

# ══════════════════ SLIDE 9 — XGB vs FT + DECISION ══════════════════
s = slide(); header(s, "FT-Transformer vs. XGBoost — and the Deployment Call",
                  "SEMESTER 1 · MODEL SELECTION")
figure(s, f"{FIG}/xgb_vs_ft_final.png", 0.55, 1.5, 6.4, 3.0)
text(s, Inches(0.8), Inches(4.7), Inches(6.0), Inches(1.9),
     [[("FT wins 3 of 4 tasks (mean 0.931 vs 0.927), largest gain on "
        "heart (+0.012).", 14, INK, False)],
      [("Masked loss trains all four heads even with one label per row.",
        13, MUTED, False)]], space_after=8)
box(s, Inches(7.25), Inches(1.5), Inches(5.45), Inches(5.25))
text(s, Inches(7.6), Inches(1.8), Inches(4.8), Inches(4.7),
     [[("WHY XGBOOST DEPLOYED", 13, TEAL, True)],
      [("✓ Within ~0.013 of the per-disease best on every task — "
        "consistency enables one shared stack", 14, INK, False)],
      [("✓ <10 ms inference, .pkl artifacts, no GPU", 14, INK, False)],
      [("✓ Exact TreeSHAP attributions — no surrogate approximation",
        14, INK, False)],
      [("✓ FT runs produced inconsistent artifacts — kept as the "
        "research/DL deliverable, not the unreliable production head",
        14, INK, False)],
      [(" ", 6, MUTED, False)],
      [("“Deep learning is enhancement, not requirement” — as scoped in "
        "the proposal.", 13, MUTED, False)]], space_after=12)
footer(s, 9)

# ══════════════════ SLIDE 10 — ARCHITECTURE ══════════════════
s = slide(); header(s, "Production Architecture — One Engine, Three Surfaces",
                  "SEMESTER 2 · SYSTEM")
figure(s, f"{FIG}/sem2_architecture.png", 0.55, 1.5, 9.1, 5.35)
bullets(s, Inches(9.9), Inches(1.7), Inches(3.0), Inches(5), [
    ("Shared engine ", "→ Streamlit, FastAPI, PDF all call it"),
    ("Byte-identical ", "predictions across surfaces (verified 10⁻⁶)"),
    ("7 ms ", "/predict latency vs 500 ms target"),
    ("NLP & LLM ", "enrich explanations — never model inputs"),
], size=14, gap=11)
footer(s, 10)

# ══════════════════ SLIDE 11 — APP DEMO ══════════════════
s = slide(); header(s, "The Application", "SEMESTER 2 · DEMO")
box(s, Inches(0.55), Inches(1.5), Inches(8.2), Inches(5.3),
    color=RGBColor(0x11, 0x11, 0x11), line=TEAL)
s.shapes.add_picture(f"{FIG}/app_screenshot_current.png",
                     Inches(0.62), Inches(1.57), Inches(8.06), Inches(5.16))
bullets(s, Inches(9.0), Inches(1.6), Inches(3.75), Inches(5), [
    ("Panel checkboxes ", "— unknown labs stay absent, never forced"),
    ("Risk cards ", "with marker-coverage chips (n/m)"),
    ("SHAP tabs ", "per disease + cross-disease expander"),
    ("PDF download ", "— 6-page patient report"),
    ("AI explanations + chat ", "with disease-aware routing"),
], size=14, gap=11)
footer(s, 11)

# ══════════════════ SLIDE 12 — SHAP ══════════════════
s = slide(); header(s, "Explainability — Per-Patient, Not Per-Model",
                  "SEMESTER 2 · TRUST")
figure(s, f"{FIG}/shap_diverging_caseA.png", 0.55, 1.5, 7.6, 4.5)
bullets(s, Inches(8.5), Inches(1.7), Inches(4.2), Inches(4.4), [
    ("TreeSHAP ", "on each patient vector — exact, no surrogate"),
    ("Log-odds contributions ", "sum to the patient's own logit"),
    ("Verified per-instance: ", "diabetic +12.09 vs healthy −2.77; "
     "drivers reorder per patient"),
    ("Feeds everything: ", "app charts, PDF pages, LLM payload"),
], size=14, gap=12)
footer(s, 12)

# ══════════════════ SLIDE 13 — CROSS-DISEASE CENTERPIECE ══════════════════
s = slide(); header(s, "Cross-Disease Signal — The Thesis Evidence",
                  "SEMESTER 2 · CORE CONTRIBUTION")
figure(s, f"{FIG}/cross_disease_attribution_caseA.png", 0.55, 1.45, 6.5, 2.75)
figure(s, f"{FIG}/panel_ablation_caseA.png", 0.55, 4.35, 6.5, 2.5)
bullets(s, Inches(7.4), Inches(1.6), Inches(5.35), Inches(5.2), [
    ("Attribution: ", "a diabetic patient with no kidney markers — the "
     "kidney score's largest driver is the diabetes family (+0.63 "
     "log-odds)"),
    ("Ablation: ", "diabetes 100% → 18.9% without its panel "
     "(panel-driven); kidney 96.7% → 95.7% (cross-disease-driven)"),
    ("Honesty gate: ", "marker-less scores are flagged “cross-disease "
     "estimate” with the exact markers to provide"),
    ("Found in testing: ", "sentinel extrapolation on absent panels — "
     "surfaced as a limitation, not hidden"),
], size=15, gap=14)
footer(s, 13)

# ══════════════════ SLIDE 14 — INTERRELATION + HONESTY ══════════════════
s = slide(); header(s, "Dataset-Level Transfer — Reported Honestly",
                  "SEMESTER 2 · VALIDATION")
figure(s, f"{FIG}/interrelation_stress_test.png", 0.55, 1.5, 7.3, 3.1)
text(s, Inches(0.8), Inches(4.85), Inches(7.0), Inches(1.8),
     [[("3 of 5 scenario pairs confirmed:", 15, INK, True)],
      [("✓ Kidney→Heart +0.122   ✓ Liver→Diabetes +0.131   "
        "✓ Liver→Heart +0.289", 15, GREEN, True)],
      [("✗ Diabetes→Kidney and Diabetes→Heart not detected at "
        "dataset level", 14, AMBER, False)],
      [("We report the negative results — transfer is real but not "
        "universal.", 13, MUTED, False)]], space_after=8)
box(s, Inches(8.35), Inches(1.5), Inches(4.35), Inches(4.9))
text(s, Inches(8.7), Inches(1.8), Inches(3.7), Inches(4.4),
     [[("ALSO VERIFIED", 12, TEAL, True)],
      [("19/19 pytest cases green", 14, INK, False)],
      [("API = engine parity (10⁻⁶)", 14, INK, False)],
      [("10 edge cases bounded, sane directions", 14, INK, False)],
      [("10/10 NLP phrasings correct", 14, INK, False)],
      [("PDF: 6 pages, gating boxes correct", 14, INK, False)],
      [("LLM live call ~1 s, structured JSON", 14, INK, False)]],
     space_after=11)
footer(s, 14)

# ══════════════════ SLIDE 15 — CLOSE ══════════════════
s = slide(); header(s, "Limitations · Future Work · Summary", "CLOSING")
bullets(s, Inches(0.8), Inches(1.6), Inches(5.9), Inches(4.8), [
    ("Heart AUC 0.801 < 0.85 target ", "— weakest label set, 29% coverage"),
    ("Marker-less estimates ", "mix real transfer with sentinel "
     "extrapolation (flagged, not hidden)"),
    ("Kidney cohort small ", "(~2K rows); no clinical calibration or "
     "external validation"),
    ("Next: ", "missingness-aware retraining, calibration, PubMedBERT, "
     "RAG guidelines, cloud deploy"),
], size=15, gap=12)
box(s, Inches(7.15), Inches(1.6), Inches(5.55), Inches(4.7), color=CARD,
    line=TEAL)
text(s, Inches(7.5), Inches(1.95), Inches(4.9), Inches(4.2),
     [[("SUMMARY", 12, TEAL, True)],
      [("Fused 4 cohorts → 191K rows → one schema, missingness as "
        "information", 15, INK, False)],
      [("5 models benchmarked; XGBoost deployed for consistency + exact "
        "SHAP", 15, INK, False)],
      [("One engine → app + API + 6-page PDF", 15, INK, False)],
      [("Cross-disease signal measured, not claimed — attribution, "
        "ablation, 3/5 stress-test confirmations", 15, INK, False)],
      [(" ", 8, MUTED, False)],
      [("Thank you — Questions?", 22, TEAL, True)]], space_after=11)
footer(s, 15)

prs.save(OUT)
print(f"saved {OUT} — {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
