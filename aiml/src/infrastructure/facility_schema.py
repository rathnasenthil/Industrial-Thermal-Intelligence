"""
Canonical facility schema for GIFT Stage I.1.

Defines the controlled vocabulary for `facility_type`, the canonical
output column order, and deterministic facility-ID generation.
Stage I.2 (thermal-event-to-facility association, not implemented here)
will consume exactly this schema, so it is deliberately kept stable and
self-describing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# --------------------------------------------------------------------------
# Controlled vocabulary for `facility_type`.
#
# UNKNOWN and OTHER_INDUSTRIAL are legitimate, *expected* outcomes for OSM
# objects that cannot be confidently mapped to a specific category — they
# are not errors and are not evidence of a data-quality problem. Most raw
# OSM extracts around an industrial area also contain many objects (roads,
# amenities, plain buildings) that are correctly UNKNOWN.
# --------------------------------------------------------------------------
REFINERY = "REFINERY"
POWER_PLANT = "POWER_PLANT"
MINE = "MINE"
INDUSTRIAL_AREA = "INDUSTRIAL_AREA"
LNG_TERMINAL = "LNG_TERMINAL"
OTHER_INDUSTRIAL = "OTHER_INDUSTRIAL"
UNKNOWN = "UNKNOWN"

FACILITY_TYPES: tuple[str, ...] = (
    REFINERY,
    POWER_PLANT,
    MINE,
    INDUSTRIAL_AREA,
    LNG_TERMINAL,
    OTHER_INDUSTRIAL,
    UNKNOWN,
)

# Geometry types this stage knows how to persist/validate. Anything else
# (e.g. LineString, GeometryCollection) is flagged as invalid rather than
# silently coerced.
SUPPORTED_GEOMETRY_TYPES: tuple[str, ...] = ("Point", "Polygon", "MultiPolygon")

VALID_OSM_TYPES: tuple[str, ...] = ("node", "way", "relation")

# Canonical output column order for osm_facilities.csv / osm_facilities.geojson.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "facility_id",
    "facility_name",
    "facility_type",
    "industrial_subtype",
    "operator",
    "landuse",
    "power_type",
    "man_made_type",
    "confidence",
    "geometry_type",
    "latitude",
    "longitude",
    "geometry_wkt",
    "osm_id",
    "osm_type",
    "osm_tags",
    "source",
    "source_version",
)


def make_osm_facility_id(osm_type: str | None, osm_id: str | None) -> str | None:
    """Deterministic facility ID derived from a stable OSM identifier.

    Args:
        osm_type: One of "node"/"way"/"relation" (case-insensitive), or
            None/blank if unknown.
        osm_id: The OSM element's numeric id (as a string), or
            None/blank if unknown.

    Returns:
        e.g. ``"osm_way_123456789"``. ``None`` if either input is
        missing/blank/unrecognized — callers should fall back to
        :func:`make_fallback_facility_id` in that case.
    """
    if not osm_type or not osm_id:
        return None
    normalized_type = str(osm_type).strip().lower()
    if normalized_type not in VALID_OSM_TYPES:
        return None
    normalized_id = str(osm_id).strip()
    if not normalized_id or normalized_id.lower() == "nan":
        return None
    return f"osm_{normalized_type}_{normalized_id}"


def make_fallback_facility_id(
    *,
    latitude: float | None,
    longitude: float | None,
    name: str | None,
    tags_json: str | None,
) -> str:
    """Deterministic fallback ID for records lacking a stable OSM id.

    KNOWN LIMITATION (documented deliberately, not hidden): this ID is
    derived only from the record's own content (rounded coordinates,
    name, tags), so it is NOT guaranteed globally unique the way a real
    OSM id is — two genuinely distinct facilities that happen to share
    identical coordinates/name/tags in the input extract would collide
    onto the same fallback id and be treated as duplicates of each other.
    Production OSM extracts should always carry a real `osm_id`/`osm_type`
    to avoid relying on this fallback.

    Args:
        latitude: Representative-point latitude (may be None).
        longitude: Representative-point longitude (may be None).
        name: Facility name, if any.
        tags_json: JSON-encoded original tags (see `encode_tags`), if any.

    Returns:
        A stable ``"fallback_<16-hex-char-digest>"`` string — deterministic
        for identical inputs on any machine, any run.
    """
    lat_key = "" if latitude is None else f"{float(latitude):.6f}"
    lon_key = "" if longitude is None else f"{float(longitude):.6f}"
    key = "|".join([lat_key, lon_key, name or "", tags_json or ""])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"fallback_{digest}"


def encode_tags(tags: dict[str, Any] | None) -> str:
    """JSON-encode an OSM tags dict deterministically (sorted keys).

    Sorting keys guarantees the same tags dict always encodes to the same
    string, which matters both for `make_fallback_facility_id` and for
    reproducible CSV output.
    """
    if not tags:
        return "{}"
    return json.dumps(tags, sort_keys=True, ensure_ascii=False, default=str)


def decode_tags(tags_json: str | None) -> dict[str, Any]:
    """Inverse of :func:`encode_tags`; returns ``{}`` for null/blank input."""
    if not tags_json:
        return {}
    try:
        decoded = json.loads(tags_json)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}
