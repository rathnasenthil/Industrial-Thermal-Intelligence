"""
Canonicalize null-ish values without a fragile pandas dependency path.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

# Exact fields included in the hash payload, in this order.
# Aligned with aiml/src/preprocessing/duplicates.py exact-duplicate column set.
OBSERVATION_HASH_FIELDS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
    "type",
)

_HASH_VERSION = "v1"


def canonicalize_hash_value(value: Any) -> str:
    """
    Canonical string form for one hash field.

    Rules:
    - None / NaN / pandas NA → ""
    - strings are stripped; empty after strip → ""
    - literal 'nan' / 'none' / '<na>' / 'nat' (case-insensitive) → ""
    - other values → strip(str(value))
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    # Handle pandas NA / NaT when pandas is available.
    try:
        import pandas as pd

        result = pd.isna(value)
        if result is True:
            return ""
    except Exception:  # noqa: BLE001
        pass

    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "<na>", "nat"}:
        return ""
    return text


def build_observation_hash_payload(row: Mapping[str, Any]) -> str:
    """Build the canonical pre-hash payload (useful for debugging/tests)."""
    lines = [f"hash_version={_HASH_VERSION}"]
    for field in OBSERVATION_HASH_FIELDS:
        lines.append(f"{field}={canonicalize_hash_value(row.get(field))}")
    return "\n".join(lines)


def compute_observation_hash(row: Mapping[str, Any]) -> str:
    """
    Return SHA-256 hex digest for a FIRMS observation row/mapping.

    The same logical observation must always produce the same hash across
    polling cycles when native field values are unchanged.
    """
    payload = build_observation_hash_payload(row)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
