"""Normalize independent reference labels into a conservative vocabulary."""

from __future__ import annotations

import pandas as pd

from src.validation.config import (
    LABEL_AGRICULTURAL,
    LABEL_AMBIGUOUS,
    LABEL_INDUSTRIAL,
    LABEL_NATURAL,
    LABEL_OTHER,
    LABEL_UNKNOWN,
)
from src.validation.validation_schema import clean_text

_RAW_TO_NORMALIZED: dict[str, str] = {
    "industrial": LABEL_INDUSTRIAL,
    "industrial_fire": LABEL_INDUSTRIAL,
    "industry": LABEL_INDUSTRIAL,
    "factory": LABEL_INDUSTRIAL,
    "power_plant": LABEL_INDUSTRIAL,
    "gas_flare": LABEL_INDUSTRIAL,
    "flare": LABEL_INDUSTRIAL,
    "natural": LABEL_NATURAL,
    "wildfire": LABEL_NATURAL,
    "forest_fire": LABEL_NATURAL,
    "vegetation": LABEL_NATURAL,
    "bushfire": LABEL_NATURAL,
    "agricultural": LABEL_AGRICULTURAL,
    "agriculture": LABEL_AGRICULTURAL,
    "crop": LABEL_AGRICULTURAL,
    "stubble": LABEL_AGRICULTURAL,
    "ag_burn": LABEL_AGRICULTURAL,
    "other": LABEL_OTHER,
    "unknown": LABEL_UNKNOWN,
    "ambiguous": LABEL_AMBIGUOUS,
    "uncertain": LABEL_AMBIGUOUS,
}


def normalize_label(raw: str | None) -> str:
    """Map a raw reference label to the Stage V vocabulary."""
    text = clean_text(raw)
    if text is None:
        return LABEL_UNKNOWN
    key = text.lower().replace("-", "_").replace(" ", "_")
    if key in _RAW_TO_NORMALIZED:
        return _RAW_TO_NORMALIZED[key]
    # Partial contains
    for token, label in _RAW_TO_NORMALIZED.items():
        if token in key:
            return label
    return LABEL_UNKNOWN


def normalize_reference_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure reference_label_raw / reference_label_normalized exist."""
    out = df.copy()
    if "reference_label_raw" not in out.columns:
        out["reference_label_raw"] = None
    out["reference_label_raw"] = out["reference_label_raw"].map(lambda v: clean_text(v))
    if (
        "reference_label_normalized" in out.columns
        and out["reference_label_normalized"].notna().any()
    ):
        out["reference_label_normalized"] = out["reference_label_normalized"].map(
            lambda v: normalize_label(v) if clean_text(v) else LABEL_UNKNOWN
        )
    else:
        out["reference_label_normalized"] = out["reference_label_raw"].map(normalize_label)
    return out
