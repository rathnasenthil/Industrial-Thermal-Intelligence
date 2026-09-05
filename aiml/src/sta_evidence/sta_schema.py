"""
Canonical NASA STA schema and deterministic ID helpers for Stage I.5.

STA is experimental/provisional NASA FIRMS evidence — not ground truth and
not a source-classification label.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.sta_evidence.config import LAYER_DETECTION, LAYER_MASK

CANONICAL_COLUMNS: tuple[str, ...] = (
    "sta_id",
    "sta_layer_type",
    "geometry_type",
    "latitude",
    "longitude",
    "geometry_wkt",
    "sta_source",
    "sta_source_version",
    "sta_source_url",
    "sta_download_date",
    "sta_layer",
    "source_date",
    "observation_datetime",
    "raw_attributes",
    "is_valid",
    "rejection_reason",
)

SUPPORTED_GEOMETRY_TYPES: frozenset[str] = frozenset(
    {"Point", "Polygon", "MultiPolygon", "MultiPoint", "LineString", "MultiLineString"}
)

VALID_MATCH_GEOMETRY_TYPES: frozenset[str] = frozenset({"Point", "Polygon", "MultiPolygon", "MultiPoint"})


def deterministic_sta_id(layer_type: str, native_id: str | None, geometry_wkt: str | None, index: int) -> str:
    """Build a stable STA identifier.

    Prefer a NASA-native ID when present; otherwise hash layer + WKT + index.
    Never uses random UUIDs.
    """
    layer = str(layer_type).upper()
    if native_id is not None and str(native_id).strip() and str(native_id).lower() != "nan":
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(native_id).strip())
        return f"sta_{layer.lower()}_{safe}"
    payload = f"{layer}|{geometry_wkt or ''}|{index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"sta_{layer.lower()}_hash_{digest}"


def empty_canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CANONICAL_COLUMNS))


def serialize_raw_attributes(attrs: dict[str, Any] | None) -> str | None:
    if not attrs:
        return None
    # Deterministic key order; drop null-like values; never emit literal "nan".
    parts: list[str] = []
    for key in sorted(attrs.keys()):
        value = attrs[key]
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            continue
        parts.append(f"{key}={text}")
    return ";".join(parts) if parts else None
