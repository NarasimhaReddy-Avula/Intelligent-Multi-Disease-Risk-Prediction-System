from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.guidance.rules import GuidanceBundle, get_guidance_for_disease, get_overall_guidance
from src.llm.clients import (
    LLMError,
    OpenAICompatibleConfig,
    load_openai_compatible_config_from_env,
    openai_compatible_chat,
)


DISCLAIMER = (
    "This information is for education/research only and is not a medical diagnosis. "
    "For personal medical advice, please consult a qualified clinician. "
    "If you have urgent or severe symptoms, seek emergency care."
)

# Recommended default for a free online tier (can be overridden via env vars).
# Groq retired llama-3.1-8b-instant (Aug 2026); gpt-oss-20b is the free-tier replacement.
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def _compact_factor_lines(factors: List[Dict[str, Any]], *, max_items: int = 5) -> List[str]:
    lines: List[str] = []
    for item in factors[: int(max_items)]:
        label = str(item.get("label") or item.get("feature"))
        val = str(item.get("raw_value_display") or "")
        status = str(item.get("status") or "")
        strength = str(item.get("impact_strength") or "")
        if status and status not in {"unknown"}:
            lines.append(f"{label}: {val} ({status}, {strength} impact)")
        else:
            lines.append(f"{label}: {val} ({strength} impact)")
    return lines


def build_patient_payload_for_llm(
    *,
    disease_name: str,
    probability: float,
    risk_level: str,
    top_increasing: List[Dict[str, Any]],
    top_decreasing: List[Dict[str, Any]],
    missing_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a compact, model-safe payload for an online LLM."""

    return {
        "disease": disease_name,
        "risk": {
            "probability": round(float(probability), 6),
            "percent": round(float(probability) * 100.0, 1),
            "level": str(risk_level),
        },
        "top_factors": {
            "increasing": top_increasing,
            "decreasing": top_decreasing,
        },
        "missing_features": list(missing_features or []),
        "constraints": {
            "no_diagnosis": True,
            "no_prescriptions": True,
            "no_dosing": True,
            "medication_classes_ok": True,
        },
    }


def build_overall_payload_for_llm(
    *,
    diseases: List[Dict[str, Any]],
    missing_features: Optional[List[str]] = None,
    patient_reported: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a multi-disease payload. Each entry of `diseases` is:
    {"name", "target", "probability", "level", "top_increasing", "top_decreasing"}"""

    return {
        "overall": True,
        "disease": "Overall multi-disease assessment (diabetes, heart, liver, kidney)",
        "diseases": diseases,
        "missing_features": list(missing_features or []),
        "patient_reported": dict(patient_reported or {}),
        "constraints": {
            "no_diagnosis": True,
            "no_prescriptions": True,
            "no_dosing": True,
            "medication_classes_ok": True,
        },
    }


def _system_prompt() -> str:
    return (
        "You are a cautious health education assistant embedded in a research app. "
        "You must follow these rules:\n"
        "- Do NOT diagnose.\n"
        "- Do NOT prescribe or tell the user to start/stop a medication.\n"
        "- Do NOT provide dosages.\n"
        "- You MAY mention common medication classes that clinicians often consider for the condition, framed as discussion topics.\n"
        "- Do NOT claim certainty; use neutral language.\n"
        "- Use only the provided factors; do not invent labs or history.\n"
        "- Always include a short disclaimer.\n\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        "  \"summary\": string,\n"
        "  \"key_factors_increasing\": [string],\n"
        "  \"key_factors_decreasing\": [string],\n"
        "  \"what_to_do_next\": [string],\n"
        "  \"when_to_seek_care\": [string],\n"
        "  \"disclaimer\": string\n"
        "}"
    )


def _overall_factor_lines(payload: Dict[str, Any], *, per_disease: int = 3) -> tuple:
    """Human-readable factor lines across all diseases, prefixed with disease name."""
    inc_lines: List[str] = []
    dec_lines: List[str] = []
    for d in payload.get("diseases", []):
        name = d.get("name", "?")
        for item in _compact_factor_lines(d.get("top_increasing", []), max_items=per_disease):
            inc_lines.append(f"{name} — {item}")
        for item in _compact_factor_lines(d.get("top_decreasing", []), max_items=per_disease):
            dec_lines.append(f"{name} — {item}")
    return inc_lines, dec_lines


def _build_messages(payload: Dict[str, Any], guidance: GuidanceBundle) -> List[Dict[str, str]]:
    if payload.get("overall"):
        inc_lines, dec_lines = _overall_factor_lines(payload)
    else:
        inc_lines = _compact_factor_lines(payload.get("top_factors", {}).get("increasing", []))
        dec_lines = _compact_factor_lines(payload.get("top_factors", {}).get("decreasing", []))

    user_content = {
        "patient_payload": payload,
        "rule_based": {
            "doctor_to_consult": guidance.doctor_to_consult,
            "lifestyle_tips": guidance.lifestyle_tips,
            "diet_tips": guidance.diet_tips,
                "diet_template": guidance.diet_template,
                "workup_tips": guidance.workup_tips,
                "medication_topics": guidance.medication_topics,
            "red_flags": guidance.red_flags,
        },
        "human_readable_factors": {
            "increasing": inc_lines,
            "decreasing": dec_lines,
        },
    }

    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


def _safe_default_response(payload: Dict[str, Any], guidance: GuidanceBundle) -> Dict[str, Any]:
    if payload.get("overall"):
        inc_lines, dec_lines = _overall_factor_lines(payload)
        ranked = sorted(payload.get("diseases", []),
                        key=lambda d: float(d.get("probability", 0.0)), reverse=True)
        summary_parts = "; ".join(
            f"{d.get('name')} {round(float(d.get('probability', 0.0)) * 100, 1)}% ({d.get('level')})"
            for d in ranked
        )
        summary = (
            f"Overall model risk profile — {summary_parts}. "
            "These are model outputs, not diagnoses."
        )
    else:
        inc_lines = _compact_factor_lines(payload.get("top_factors", {}).get("increasing", []))
        dec_lines = _compact_factor_lines(payload.get("top_factors", {}).get("decreasing", []))
        summary = (
            f"The model estimates {payload['risk']['percent']}% risk for {payload['disease']} "
            f"({payload['risk']['level']} risk). This is a model output, not a diagnosis."
        )

    return {
        "summary": summary,
        "key_factors_increasing": inc_lines,
        "key_factors_decreasing": dec_lines,
        "what_to_do_next": (
            [f"Consider discussing results with: {guidance.doctor_to_consult}."]
            + guidance.workup_tips[:3]
            + guidance.diet_template[:3]
            + (
                [
                    "Medication discussion topics (with a clinician; not a prescription): "
                    + "; ".join(guidance.medication_topics[:3])
                ]
                if guidance.medication_topics
                else []
            )
            + guidance.lifestyle_tips[:3]
            + guidance.diet_tips[:3]
        ),
        "when_to_seek_care": guidance.red_flags,
        "disclaimer": DISCLAIMER,
    }


def generate_ai_explanation(
    *,
    payload: Dict[str, Any],
    cfg: Optional[OpenAICompatibleConfig] = None,
) -> Dict[str, Any]:
    """Generate a patient-friendly explanation.

    If no config is provided (or env vars are missing), returns a rule-based fallback.
    """

    if payload.get("overall"):
        guidance = get_overall_guidance(
            {str(d.get("name")): str(d.get("level")) for d in payload.get("diseases", [])}
        )
    else:
        guidance = get_guidance_for_disease(
            disease=str(payload.get("disease", "")),
            risk_level=str(payload.get("risk", {}).get("level", "")),
        )

    if cfg is None:
        cfg = load_openai_compatible_config_from_env(
            default_base_url=DEFAULT_BASE_URL,
            default_model=DEFAULT_MODEL,
        )

    if cfg is None:
        return {"ok": True, "mode": "rules", "data": _safe_default_response(payload, guidance)}

    def _augment_explanation_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure key fields exist and add disease-specific next steps if missing/too short."""

        default = _safe_default_response(payload, guidance)

        # Ensure list fields are present and well-typed.
        for k in ["key_factors_increasing", "key_factors_decreasing", "what_to_do_next", "when_to_seek_care"]:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                data[k] = [v.strip()]
            elif not isinstance(v, list):
                data[k] = list(default.get(k) or [])

        # If the model returned very short next steps, enrich with disease-specific guidance.
        wtdn = data.get("what_to_do_next")
        if not isinstance(wtdn, list):
            wtdn = []

        if len(wtdn) < 5:
            data["what_to_do_next"] = list(default.get("what_to_do_next") or [])
            return data

        # Otherwise, ensure at least one item from each disease-specific bucket is present.
        to_add: List[str] = []

        if guidance.workup_tips:
            to_add.extend(guidance.workup_tips[:2])
        if guidance.diet_template:
            to_add.extend(guidance.diet_template[:2])
        if guidance.medication_topics:
            to_add.append(
                "Medication discussion topics (with a clinician; not a prescription): "
                + "; ".join(guidance.medication_topics[:3])
            )

        existing_text = "\n".join(str(x) for x in wtdn)
        for item in to_add:
            if item and item[:40] not in existing_text:
                wtdn.append(item)
        data["what_to_do_next"] = wtdn

        return data

    try:
        messages = _build_messages(payload, guidance)
        resp = openai_compatible_chat(cfg=cfg, messages=messages, temperature=0.2, max_tokens=900,
                                      response_format_json=True, reasoning_effort="low")
        data = resp.get("json")
        if not isinstance(data, dict):
            return {"ok": True, "mode": "llm_text", "data": {"summary": resp.get("text", ""), "disclaimer": DISCLAIMER}}

        data = _augment_explanation_json(data)

        # Ensure disclaimer always present.
        if not data.get("disclaimer"):
            data["disclaimer"] = DISCLAIMER

        return {"ok": True, "mode": "llm_json", "data": data, "latency_s": resp.get("latency_s")}
    except LLMError as e:
        return {"ok": False, "mode": "error", "error": str(e), "data": _safe_default_response(payload, guidance)}
    except Exception as e:
        return {"ok": False, "mode": "error", "error": str(e), "data": _safe_default_response(payload, guidance)}


def generate_ai_chat_reply(
    *,
    payload: Dict[str, Any],
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    cfg: Optional[OpenAICompatibleConfig] = None,
    max_history_messages: int = 8,
) -> Dict[str, Any]:
    """Generate a follow-up chat reply grounded in the patient payload."""

    if payload.get("overall"):
        guidance = get_overall_guidance(
            {str(d.get("name")): str(d.get("level")) for d in payload.get("diseases", [])}
        )
    else:
        guidance = get_guidance_for_disease(
            disease=str(payload.get("disease", "")),
            risk_level=str(payload.get("risk", {}).get("level", "")),
        )

    if cfg is None:
        cfg = load_openai_compatible_config_from_env(
            default_base_url=DEFAULT_BASE_URL,
            default_model=DEFAULT_MODEL,
        )

    if cfg is None:
        # Minimal safe fallback
        return {
            "ok": True,
            "mode": "rules",
            "answer": (
                "I can answer general questions about the model outputs, but no LLM API key is configured. "
                "Based on the shown factors, a reasonable next step is to discuss the results with a clinician and "
                "focus on general lifestyle basics (activity, sleep, balanced diet). "
                + f"\n\nDisclaimer: {DISCLAIMER}"
            ),
        }

    system = (
        "You are a cautious health education assistant. Rules:\n"
        "- Do NOT diagnose.\n"
        "- Do NOT prescribe or tell the user to start/stop a medication.\n"
        "- Do NOT provide dosages.\n"
        "- You MAY mention common medication classes clinicians consider, framed as discussion topics.\n"
        "- Use ONLY the provided patient payload and rule-based tips; do not invent labs/history.\n"
        "- Answer for the disease named in the payload (do not switch diseases unless explicitly asked and supported).\n"
        "- If a factor includes `status` and/or `normal_range`, do NOT label a value high/low unless the status indicates that.\n"
        "- Provide a helpful, moderately detailed answer (typically 6–12 lines).\n"
        "- Use Markdown bullets for lists.\n"
        "- When relevant, include: risk context (percent + level), key factors, and next steps including diet template, workup, and clinician discussion topics.\n"
        "Return ONLY JSON: {\"answer\": string, \"disclaimer\": string}."
    )

    if payload.get("overall"):
        inc_lines, dec_lines = _overall_factor_lines(payload)
    else:
        inc_lines = _compact_factor_lines(payload.get("top_factors", {}).get("increasing", []))
        dec_lines = _compact_factor_lines(payload.get("top_factors", {}).get("decreasing", []))

    def _augment_if_too_short(ans: str) -> str:
        a = (ans or "").strip()
        # If the model answers too briefly, expand with grounded context.
        if len(a) >= 240 and a.count("\n") >= 2:
            return a

        missing = payload.get("missing_features") if isinstance(payload.get("missing_features"), list) else []

        extra: List[str] = []
        if payload.get("overall"):
            ranked = sorted(payload.get("diseases", []),
                            key=lambda d: float(d.get("probability", 0.0)), reverse=True)
            ctx_lines = "; ".join(
                f"{d.get('name')} {round(float(d.get('probability', 0.0)) * 100, 1)}% ({d.get('level')})"
                for d in ranked
            )
            if ctx_lines:
                extra.append(f"**Overall risk context**: {ctx_lines}")
        else:
            disease = str(payload.get("disease") or "")
            risk = payload.get("risk", {}) if isinstance(payload.get("risk"), dict) else {}
            pct = risk.get("percent")
            lvl = risk.get("level")
            if disease and pct is not None and lvl is not None:
                extra.append(f"**Risk context ({disease})**: {pct}% ({lvl}).")

        if inc_lines:
            extra.append("**Top factors increasing risk**")
            extra.extend([f"- {s}" for s in inc_lines[:3]])

        if dec_lines:
            extra.append("**Top factors decreasing risk**")
            extra.extend([f"- {s}" for s in dec_lines[:2]])

        if missing:
            shown = missing[:10]
            tail = " …" if len(missing) > 10 else ""
            extra.append("**Missing/auto-handled inputs**")
            extra.append("- " + ", ".join(str(x) for x in shown) + tail)

        next_steps: List[str] = []
        next_steps.extend(guidance.lifestyle_tips[:2])
        next_steps.extend(guidance.diet_tips[:2])
        if next_steps:
            extra.append("**Practical next steps (general)**")
            extra.extend([f"- {s}" for s in next_steps if s])

        if guidance.diet_template:
            extra.append("**Diet template (general)**")
            extra.extend([f"- {s}" for s in guidance.diet_template[:3] if s])

        if guidance.workup_tips:
            extra.append("**Clinical follow-up / workup topics**")
            extra.extend([f"- {s}" for s in guidance.workup_tips[:3] if s])

        if guidance.medication_topics:
            extra.append("**Medication classes to discuss with a clinician (not a prescription)**")
            extra.extend([f"- {s}" for s in guidance.medication_topics[:3] if s])

        if guidance.red_flags:
            extra.append("**When to seek care (red flags)**")
            extra.extend([f"- {s}" for s in guidance.red_flags[:3]])

        if not extra:
            return a

        return (a + "\n\n" if a else "") + "\n".join(extra)

    ctx = {
        "patient_payload": payload,
        "rule_based": {
            "doctor_to_consult": guidance.doctor_to_consult,
            "lifestyle_tips": guidance.lifestyle_tips,
            "diet_tips": guidance.diet_tips,
            "diet_template": guidance.diet_template,
            "workup_tips": guidance.workup_tips,
            "medication_topics": guidance.medication_topics,
            "red_flags": guidance.red_flags,
        },
        "human_readable_factors": {
            "increasing": inc_lines,
            "decreasing": dec_lines,
        },
    }

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": json.dumps(ctx, ensure_ascii=False)})

    # Include brief history, if any.
    if chat_history:
        for m in chat_history[-int(max_history_messages) :]:
            if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        resp = openai_compatible_chat(cfg=cfg, messages=messages, temperature=0.2, max_tokens=900,
                                      response_format_json=True, reasoning_effort="low")
        data = resp.get("json")
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            data["answer"] = _augment_if_too_short(data.get("answer", ""))
            if not data.get("disclaimer"):
                data["disclaimer"] = DISCLAIMER
            return {"ok": True, "mode": "llm_json", **data}

        # Fallback to text
        return {
            "ok": True,
            "mode": "llm_text",
            "answer": _augment_if_too_short(str(resp.get("text", ""))),
            "disclaimer": DISCLAIMER,
        }
    except Exception as e:
        return {
            "ok": False,
            "mode": "error",
            "error": str(e),
            "answer": (
                "I hit an error contacting the LLM provider. You can still use the SHAP explanations above.\n\n"
                + f"Disclaimer: {DISCLAIMER}"
            ),
        }
