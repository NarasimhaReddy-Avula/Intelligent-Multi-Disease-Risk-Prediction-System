from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Config for OpenAI-compatible chat completion APIs.

    Works with providers that expose an OpenAI-style endpoint such as:
    - OpenAI (paid)
    - Groq (often free tier)
    - Together/OpenRouter/etc. (varies)

    Environment variables (recommended):
    - `BTP_LLM_BASE_URL` (e.g. https://api.groq.com/openai/v1)
    - `BTP_LLM_API_KEY`
    - `BTP_LLM_MODEL` (e.g. llama-3.1-8b-instant)
    """

    base_url: str
    api_key: str
    model: str
    timeout_s: int = 45


def load_openai_compatible_config_from_env(
    *,
    default_base_url: Optional[str] = None,
    default_model: Optional[str] = None,
) -> Optional[OpenAICompatibleConfig]:
    """Load config from environment variables.

    Returns None if required variables are missing.
    """

    base_url = os.environ.get("BTP_LLM_BASE_URL") or default_base_url
    api_key = os.environ.get("BTP_LLM_API_KEY")
    model = os.environ.get("BTP_LLM_MODEL") or default_model

    # Backwards/compat: allow common provider-specific key names.
    # This helps when users follow an existing .env template.
    if (not api_key) and base_url:
        base_url_l = base_url.lower()
        if "groq.com" in base_url_l:
            api_key = os.environ.get("GROQ_API_KEY")
        elif "openai.com" in base_url_l:
            api_key = os.environ.get("OPENAI_API_KEY")

    if not base_url or not api_key or not model:
        return None

    return OpenAICompatibleConfig(base_url=base_url.rstrip("/"), api_key=api_key, model=model)


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from a model response."""

    text = text.strip()
    parsed = _safe_json_loads(text)
    if isinstance(parsed, dict):
        return parsed

    # Common pattern: fenced code block
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            parsed = _safe_json_loads(part)
            if isinstance(parsed, dict):
                return parsed

    # Fallback: find first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = _safe_json_loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    return None


def openai_compatible_chat(
    *,
    cfg: OpenAICompatibleConfig,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 600,
    response_format_json: bool = True,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible Chat Completions endpoint.

    Returns:
        dict with keys:
        - "text": assistant text
        - "json": parsed JSON object if response_format_json and parse succeeded
        - "raw": raw response json
    """

    url = f"{cfg.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    # Some providers support this. If unsupported, they will ignore or error.
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    # Reasoning models (e.g. Groq's openai/gpt-oss-*) burn completion tokens on
    # chain-of-thought; "low" keeps budget for the actual answer.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    def _post(p: Dict[str, Any]):
        t0i = time.time()
        try:
            r = requests.post(url, headers=headers, json=p, timeout=cfg.timeout_s)
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}") from e
        return r, (time.time() - t0i)

    resp, dt = _post(payload)

    # Some providers advertise OpenAI-compat but don't support these params —
    # strip whatever it rejected and retry once.
    if resp.status_code == 400:
        payload2 = dict(payload)
        if reasoning_effort and "reasoning_effort" in payload2:
            payload2.pop("reasoning_effort", None)
        if response_format_json:
            payload2.pop("response_format", None)
        if payload2 != payload:
            resp, dt = _post(payload2)

    if resp.status_code >= 400:
        raise LLMError(f"LLM API error {resp.status_code}: {resp.text[:500]}")

    raw = resp.json()
    try:
        text = raw["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(f"Unexpected LLM response format: {raw}") from e

    out: Dict[str, Any] = {"text": text, "json": None, "raw": raw, "latency_s": dt}
    if response_format_json:
        out["json"] = _extract_first_json_object(text)

    return out
