from __future__ import annotations

import math
from pathlib import Path
import textwrap

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_figures"


FEATURES_24 = [
    "age",
    "gender",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_total",
    "glucose_fasting",
    "hdl",
    "ldl",
    "triglycerides",
    "hba1c",
    "serum_creatinine",
    "bun",
    "gfr",
    "protein_urine",
    "alt",
    "ast",
    "alp",
    "bilirubin_total",
    "bilirubin_direct",
    "albumin",
    "total_protein",
    "ag_ratio",
    "hemoglobin",
]


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fontsize: int = 10,
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="black",
        facecolor="#f7f7f7",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )


def fig_unified_schema() -> None:
    groups = {
        "Demographics": ["age", "gender"],
        "Anthropometrics": ["bmi"],
        "Vitals": ["systolic_bp", "diastolic_bp"],
        "Lipids": ["cholesterol_total", "hdl", "ldl", "triglycerides"],
        "Glucose": ["glucose_fasting", "hba1c"],
        "Kidney panel": ["serum_creatinine", "bun", "gfr", "protein_urine"],
        "Liver panel": [
            "alt",
            "ast",
            "alp",
            "bilirubin_total",
            "bilirubin_direct",
            "albumin",
            "total_protein",
            "ag_ratio",
        ],
        "Hematology": ["hemoglobin"],
    }

    fig = plt.figure(figsize=(12.5, 7))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    ax.text(
        0.5,
        1.02,
        "Unified Clinical Schema (24 Features)",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        transform=ax.transAxes,
    )

    x_left = 0.06
    x_right = 0.52
    y_top = 0.88
    box_w = 0.42
    box_h = 0.13
    gap_y = 0.035

    items = list(groups.items())
    for i, (title, feats) in enumerate(items):
        col = i % 2
        row = i // 2
        x = x_left if col == 0 else x_right
        y = y_top - row * (box_h + gap_y)

        wrapped_feats = textwrap.fill(", ".join(feats), width=44)
        text = f"{title}\n{wrapped_feats}"
        fontsize = 10 if len(feats) <= 4 else 9
        draw_box(ax, x, y, box_w, box_h, text, fontsize=fontsize)

    ax.text(
        0.5,
        0.02,
        "Features are shared across cohorts where available; structurally missing panels are encoded via masked/sentinel strategies.",
        ha="center",
        va="bottom",
        fontsize=10,
        transform=ax.transAxes,
    )

    save_fig(fig, OUT_DIR / "unified_schema_24_features.png")


def fig_data_pipeline() -> None:
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    ax.text(
        0.5,
        1.02,
        "Dataset Fusion Pipeline and Missingness Encodings",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        transform=ax.transAxes,
    )

    draw_box(ax, 0.05, 0.78, 0.25, 0.12, "Raw datasets\n(Heart, Diabetes, Liver, Kidney)")
    draw_box(ax, 0.37, 0.78, 0.26, 0.12, "Schema mapping\n+ unit alignment")
    draw_box(ax, 0.70, 0.78, 0.25, 0.12, "Unified dataset\n+ source_dataset")

    arrow(ax, 0.30, 0.84, 0.37, 0.84)
    arrow(ax, 0.63, 0.84, 0.70, 0.84)

    draw_box(ax, 0.37, 0.58, 0.26, 0.12, "QC: deduplicate\n+ type fixes")
    arrow(ax, 0.82, 0.78, 0.50, 0.70)

    draw_box(ax, 0.10, 0.35, 0.30, 0.16, "Sentinel encoding\nMissing → -999\n(Tree-friendly)")
    draw_box(ax, 0.60, 0.35, 0.30, 0.16, "Masked encoding\nValue + availability mask\n(Deep-model friendly)")

    arrow(ax, 0.50, 0.58, 0.25, 0.51)
    arrow(ax, 0.50, 0.58, 0.75, 0.51)

    draw_box(ax, 0.10, 0.15, 0.30, 0.12, "Scaling + splits\n(70/15/15)")
    draw_box(ax, 0.60, 0.15, 0.30, 0.12, "Scaling + splits\n(70/15/15)")

    arrow(ax, 0.25, 0.35, 0.25, 0.27)
    arrow(ax, 0.75, 0.35, 0.75, 0.27)

    ax.text(
        0.5,
        0.04,
        "Both encodings use identical split indices to ensure fair comparisons.",
        ha="center",
        va="bottom",
        fontsize=10,
        transform=ax.transAxes,
    )

    save_fig(fig, OUT_DIR / "data_pipeline_masked_sentinel.png")


def fig_system_architecture() -> None:
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    ax.text(
        0.5,
        1.02,
        "End-to-End System Architecture",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        transform=ax.transAxes,
    )

    draw_box(ax, 0.05, 0.72, 0.26, 0.14, "Unified dataset\n(sentinel/masked)")
    draw_box(ax, 0.37, 0.72, 0.26, 0.14, "Training + evaluation\n(Week 6/7)")
    draw_box(ax, 0.69, 0.72, 0.26, 0.14, "Saved artifacts\nmodels/ + scalers/")

    arrow(ax, 0.31, 0.79, 0.37, 0.79)
    arrow(ax, 0.63, 0.79, 0.69, 0.79)

    draw_box(ax, 0.05, 0.42, 0.26, 0.14, "Streamlit UI\n(webapp/app.py)")
    draw_box(ax, 0.37, 0.42, 0.26, 0.14, "Prediction service\n(XGBoost per disease)")
    draw_box(ax, 0.69, 0.42, 0.26, 0.14, "XAI + LLM layer\n(SHAP + online chat)")

    arrow(ax, 0.82, 0.72, 0.18, 0.56)
    arrow(ax, 0.18, 0.42, 0.37, 0.49)
    arrow(ax, 0.63, 0.49, 0.69, 0.49)

    draw_box(ax, 0.37, 0.18, 0.26, 0.12, "User output\n(risk + factors + guidance)")
    arrow(ax, 0.82, 0.42, 0.50, 0.30)

    ax.text(
        0.5,
        0.04,
        "LLM is optional and provider-agnostic (OpenAI-compatible HTTP) via environment variables.",
        ha="center",
        va="bottom",
        fontsize=10,
        transform=ax.transAxes,
    )

    save_fig(fig, OUT_DIR / "system_architecture.png")


def fig_streamlit_layout() -> None:
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    ax.text(
        0.5,
        1.02,
        "Streamlit Application Layout (Demo)",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Outer window
    outer = patches.FancyBboxPatch(
        (0.06, 0.12),
        0.88,
        0.78,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor="#ffffff",
    )
    ax.add_patch(outer)

    # Sidebar
    draw_box(ax, 0.08, 0.16, 0.22, 0.70, "Sidebar\nDisease select\nInput form\nWhat-if controls")

    # Main panels
    draw_box(ax, 0.33, 0.64, 0.58, 0.22, "Risk scores\n(4 diseases) + calibration notes")
    draw_box(ax, 0.33, 0.36, 0.28, 0.22, "Local SHAP factors\n(top +/− features)")
    draw_box(ax, 0.63, 0.36, 0.28, 0.22, "Global importance\n(sampled reference set)")
    draw_box(ax, 0.33, 0.16, 0.58, 0.16, "AI explanation & chat\n(online LLM + rule-based fallback)")

    ax.text(
        0.5,
        0.04,
        "This figure is a layout diagram (useful for slides) when a real screenshot is not available.",
        ha="center",
        va="bottom",
        fontsize=10,
        transform=ax.transAxes,
    )

    save_fig(fig, OUT_DIR / "streamlit_app_screenshot.png")


def fig_shap_outputs() -> None:
    # We generate an example for the diabetes model because it has large sample size.
    data_path = ROOT / "datasets" / "combined_unified_sentinel_scaled.csv"
    model_path = ROOT / "models" / "ml" / "sentinel" / "diabetes_XGBoost.pkl"

    df = pd.read_csv(data_path, usecols=FEATURES_24 + ["diabetes_risk"], low_memory=False)
    df = df[df["diabetes_risk"].notna()]

    # Small sample for responsiveness
    df_sample = df.sample(250, random_state=42)
    X = df_sample[FEATURES_24]

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    top_features = [FEATURES_24[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    # Local example
    x0 = X.iloc[[0]]
    sv0 = shap_values[0]
    pred_log_odds = float(explainer.expected_value + sv0.sum())
    base_p = float(sigmoid(explainer.expected_value))
    pred_p = float(sigmoid(pred_log_odds))

    local_idx = np.argsort(np.abs(sv0))[::-1][:8]
    local_features = [FEATURES_24[i] for i in local_idx]
    local_contrib = sv0[local_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Global importance
    ax0 = axes[0]
    y = np.arange(len(top_features))[::-1]
    ax0.barh(y, top_vals[::-1], color="#4c72b0")
    ax0.set_yticks(y, labels=top_features[::-1])
    ax0.set_xlabel("Mean(|SHAP value|) on sample")
    ax0.set_title("Global feature importance (Diabetes, XGBoost)")
    ax0.grid(axis="x", alpha=0.25)

    # Local contributions
    ax1 = axes[1]
    colors = ["#55a868" if v > 0 else "#c44e52" for v in local_contrib]
    y2 = np.arange(len(local_features))[::-1]
    ax1.barh(y2, local_contrib[::-1], color=colors[::-1])
    ax1.axvline(0, color="black", lw=1)
    ax1.set_yticks(y2, labels=local_features[::-1])
    ax1.set_xlabel("SHAP contribution (log-odds)")
    ax1.set_title(f"Local explanation example\nbase p={base_p:.3f} → pred p={pred_p:.3f}")
    ax1.grid(axis="x", alpha=0.25)

    fig.suptitle("Explainability Outputs (SHAP) — Example", fontsize=16, fontweight="bold")
    fig.tight_layout()

    save_fig(fig, OUT_DIR / "shap_explainability_ui.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_unified_schema()
    fig_data_pipeline()
    fig_system_architecture()
    fig_streamlit_layout()
    fig_shap_outputs()

    print("Generated figures in:", OUT_DIR)


if __name__ == "__main__":
    main()
