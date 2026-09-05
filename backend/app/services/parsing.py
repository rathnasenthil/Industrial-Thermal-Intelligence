"""Shared CSV parsing helpers for Stage VI / I.1 / I.2 ingestion."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Optional


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def parse_optional_str(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    return str(value).strip()


def parse_optional_int(value: Any) -> Optional[int]:
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def parse_optional_bool(value: Any) -> Optional[bool]:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "t"}:
        return True
    if text in {"false", "0", "no", "f"}:
        return False
    return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    if is_missing(value):
        return None
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_json_object(value: Any) -> Optional[dict[str, Any]]:
    if is_missing(value):
        return None
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def valid_lon_lat(lon: Optional[float], lat: Optional[float]) -> bool:
    if lon is None or lat is None:
        return False
    return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0


def point_wkt(lon: float, lat: float) -> str:
    return f"POINT({lon} {lat})"
