from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureNormalRange:
    lo: float
    hi: float
    unit: str = ""


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _impact_adjective_from_share(share_pct: float) -> str:
    if share_pct >= 25.0:
        return "significant"
    if share_pct >= 12.0:
        return "moderate"
    return "small"


def _status_from_range(v: Optional[float], nr: Optional[FeatureNormalRange]) -> str:
    if v is None:
        return "missing"
    if nr is None:
        return "unknown"
    if v < nr.lo:
        return "low"
    if v > nr.hi:
        return "high"
    return "normal"


def _format_value(v: Optional[float], unit: str) -> str:
    if v is None:
        return "not provided"
    if unit:
        return f"{v:.2f} {unit}" if abs(v) < 100 else f"{v:.1f} {unit}"
    return f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"


def build_structured_shap_factors(
    *,
    feature_names: Sequence[str],
    shap_row: np.ndarray,
    patient_dict: Dict[str, Any],
    feature_labels: Dict[str, str],
    normal_ranges: Dict[str, Tuple[float, float, str]],
    top_k: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert a single-row SHAP explanation into structured top factors.

    Args:
        feature_names: model feature order
        shap_row: per-feature contribution (1D)
        patient_dict: raw (unscaled) input values
        feature_labels: mapping feature -> display label
        normal_ranges: mapping feature -> (lo, hi, unit)

    Returns:
        {"top_increasing": [...], "top_decreasing": [...]} where each item is a dict.
    """

    shap_row = np.asarray(shap_row, dtype=np.float32).reshape(-1)
    if len(feature_names) != shap_row.shape[0]:
        raise ValueError("feature_names and shap_row must have the same length")

    abs_vals = np.abs(shap_row)
    denom = float(abs_vals.sum()) or 1.0
    shares = (abs_vals / denom) * 100.0
    order = np.argsort(abs_vals)[::-1]

    inc: List[Dict[str, Any]] = []
    dec: List[Dict[str, Any]] = []

    for idx in order:
        feat = str(feature_names[int(idx)])
        contrib = float(shap_row[int(idx)])
        if contrib == 0.0:
            continue

        share_pct = float(shares[int(idx)])
        adjective = _impact_adjective_from_share(share_pct)

        raw_val_any = patient_dict.get(feat)
        raw_val = _to_float_or_none(raw_val_any)

        lo_hi_unit = normal_ranges.get(feat)
        nr = None
        if lo_hi_unit is not None:
            lo, hi, unit = lo_hi_unit
            nr = FeatureNormalRange(float(lo), float(hi), str(unit))

        status = _status_from_range(raw_val, nr)
        unit = nr.unit if nr else ""

        item = {
            "feature": feat,
            "label": feature_labels.get(feat, feat),
            "raw_value": raw_val,
            "raw_value_display": _format_value(raw_val, unit),
            "normal_range": ({"lo": nr.lo, "hi": nr.hi, "unit": nr.unit} if nr else None),
            "status": status,
            "shap_contribution": contrib,
            "impact_share_pct": round(share_pct, 1),
            "impact_strength": adjective,
            "direction": ("increase" if contrib > 0 else "decrease"),
        }

        if contrib > 0 and len(inc) < int(top_k):
            inc.append(item)
        elif contrib < 0 and len(dec) < int(top_k):
            dec.append(item)

        if len(inc) >= int(top_k) and len(dec) >= int(top_k):
            break

    return {"top_increasing": inc, "top_decreasing": dec}
