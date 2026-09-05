"""
OSM tag normalization for GIFT Stage I.1.

Two responsibilities:

1. :func:`classify_facility_type` — maps raw OSM tags to the controlled
   `facility_type` vocabulary (`src.infrastructure.facility_schema`)
   using actual, documented OSM tagging conventions as evidence. This is
   deliberately conservative: an OSM object with no recognizable
   industrial evidence becomes ``UNKNOWN``, and industrial-but-unspecific
   evidence becomes ``OTHER_INDUSTRIAL``, rather than guessing a specific
   category.
2. :func:`normalize_osm_facilities` — turns a raw loaded extract (see
   `osm_loader.py`) into the canonical facility table
   (`facility_schema.CANONICAL_COLUMNS`), computing geometry metadata,
   deterministic ids, and representative points, while preserving the
   original OSM tags verbatim.

IMPORTANT: this module produces *contextual* facility records only. It
does not, and must not, claim that a given OSM object represents ground
truth for whether a nearby thermal event is industrial — that
association (and any confidence scoring about it) belongs to Stage I.2.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import geopandas as gpd
import pandas as pd

from src.infrastructure.config import InfrastructureConfig
from src.infrastructure.facility_schema import (
    CANONICAL_COLUMNS,
    INDUSTRIAL_AREA,
    LNG_TERMINAL,
    MINE,
    OTHER_INDUSTRIAL,
    POWER_PLANT,
    REFINERY,
    SUPPORTED_GEOMETRY_TYPES,
    UNKNOWN,
    encode_tags,
    make_fallback_facility_id,
    make_osm_facility_id,
)


class FacilityTypeResult(NamedTuple):
    """Result of classifying one facility's OSM tags.

    Attributes:
        facility_type: One of `facility_schema.FACILITY_TYPES`.
        industrial_subtype: Finer-grained evidence value when available
            (e.g. the `industrial=*` value, or a `plant:source` fuel
            type), otherwise ``None``. Never invented.
        power_type: Raw `plant:source` tag value (e.g. "coal", "gas",
            "solar") when the facility matched via `power=plant`,
            otherwise ``None``.
        man_made_type: Raw `man_made` tag value, if present, regardless
            of which rule matched (kept for transparency/debugging).
        landuse: Raw `landuse` tag value, if present.
        confidence: Qualitative confidence in the mapping — "high" for a
            direct, well-established single-tag match, "medium" for a
            heuristic/combined-evidence match, ``None`` for UNKNOWN. This
            is NOT a statistical probability.
        matched_rule: Human-readable description of which rule fired, for
            auditability (e.g. ``"industrial=refinery"``).
    """

    facility_type: str
    industrial_subtype: str | None
    power_type: str | None
    man_made_type: str | None
    landuse: str | None
    confidence: str | None
    matched_rule: str | None


def _raw_tag(tags: dict[str, Any], key: str) -> str | None:
    """Original (trimmed, not lower-cased) tag value, or None if absent/blank."""
    value = tags.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _norm_tag(tags: dict[str, Any], key: str) -> str | None:
    """Lower-cased tag value, for case-insensitive rule matching only."""
    raw = _raw_tag(tags, key)
    return raw.lower() if raw is not None else None


def classify_facility_type(
    tags: dict[str, Any],
    name: str | None = None,
    lng_name_keywords: Sequence[str] = ("lng", "liquefied natural gas"),
) -> FacilityTypeResult:
    """Map raw OSM tags to the controlled `facility_type` vocabulary.

    Rules are tried in priority order (most specific/confident first),
    using real, documented OSM tagging conventions:

    * ``industrial=refinery`` -> REFINERY (documented OSM value).
    * Gas-industrial evidence (``industrial=gas`` / a gas-content storage
      tank) *combined with* an explicit "LNG" mention in the name ->
      LNG_TERMINAL. OSM has no single universal LNG-terminal tag, so this
      stage requires this extra corroborating evidence rather than
      mapping every generic gas facility to LNG_TERMINAL.
    * ``industrial=mine`` or ``landuse=quarry`` -> MINE (both documented
      OSM values for mineral extraction sites).
    * ``power=plant`` -> POWER_PLANT (documented tag for a generation
      facility; deliberately excludes ``power=substation``/``line``/
      ``tower``, which are grid infrastructure, not generation).
    * ``landuse=industrial`` -> INDUSTRIAL_AREA (generic industrial zone,
      used only when nothing more specific above matched).
    * Any other industrial-adjacent evidence (an unmapped
      ``industrial=*`` value, a generic ``man_made=works``/
      ``wastewater_plant``, or any other ``power=*`` value) ->
      OTHER_INDUSTRIAL.
    * Otherwise -> UNKNOWN. This is the expected, non-error outcome for
      the majority of ordinary OSM objects (roads, amenities, plain
      buildings) that may appear in a raw extract alongside real
      facilities.

    Args:
        tags: Raw OSM tag dict for one object (e.g. ``{"landuse":
            "industrial"}``). May or may not itself contain a ``"name"``
            key depending on the input format/loader.
        name: The facility's name, if known. Passed explicitly (rather
            than only read from `tags`) because some loaders (e.g. the
            CSV loader) surface `name` as its own column, separate from
            the freeform tags blob. Falls back to ``tags.get("name")``
            when not given.
        lng_name_keywords: See `InfrastructureConfig.lng_name_keywords`.

    Returns:
        A `FacilityTypeResult`.
    """
    industrial = _norm_tag(tags, "industrial")
    power = _norm_tag(tags, "power")
    landuse_norm = _norm_tag(tags, "landuse")
    man_made = _norm_tag(tags, "man_made")
    content = _norm_tag(tags, "content")
    name = (name if name is not None else _raw_tag(tags, "name")) or ""
    name_lower = name.lower()

    landuse_raw = _raw_tag(tags, "landuse")
    man_made_raw = _raw_tag(tags, "man_made")
    plant_source_raw = _raw_tag(tags, "plant:source")
    industrial_raw = _raw_tag(tags, "industrial")

    if industrial == "refinery":
        return FacilityTypeResult(REFINERY, industrial_raw, None, man_made_raw, landuse_raw, "high", "industrial=refinery")

    gas_evidence = industrial == "gas" or content in {"lng", "gas", "natural_gas"} or man_made in {"storage_tank", "tank"}
    name_has_lng = any(keyword.lower() in name_lower for keyword in lng_name_keywords)
    if gas_evidence and name_has_lng:
        return FacilityTypeResult(
            LNG_TERMINAL, "lng", None, man_made_raw, landuse_raw, "medium", "gas-industrial evidence + LNG name match"
        )

    if industrial == "mine":
        return FacilityTypeResult(MINE, industrial_raw, None, man_made_raw, landuse_raw, "high", "industrial=mine")
    if landuse_norm == "quarry":
        return FacilityTypeResult(MINE, "quarry", None, man_made_raw, landuse_raw, "high", "landuse=quarry")

    if power == "plant":
        return FacilityTypeResult(
            POWER_PLANT, plant_source_raw, plant_source_raw, man_made_raw, landuse_raw, "high", "power=plant"
        )

    if landuse_norm == "industrial":
        return FacilityTypeResult(INDUSTRIAL_AREA, None, None, man_made_raw, landuse_raw, "high", "landuse=industrial")

    if industrial or man_made in {"works", "wastewater_plant"} or power:
        subtype = industrial_raw or man_made_raw or _raw_tag(tags, "power")
        return FacilityTypeResult(
            OTHER_INDUSTRIAL,
            subtype,
            plant_source_raw if power else None,
            man_made_raw,
            landuse_raw,
            "medium",
            "unmapped industrial/man_made/power tag evidence",
        )

    return FacilityTypeResult(UNKNOWN, None, None, man_made_raw, landuse_raw, None, None)


def _representative_point(geometry: Any) -> tuple[float | None, float | None]:
    """Centroid of `geometry`, for spatial association only.

    NOT a replacement for the stored geometry: polygons/multipolygons
    keep their original shape in `geometry`/`geometry_wkt`; this is only
    a convenience point. Like `event_formation.geometry`'s event
    centroid, this is a simple planar centroid on WGS84 degree
    coordinates — adequate at facility scales (well under 100km), not a
    true geodesic centroid.
    """
    if geometry is None or geometry.is_empty:
        return None, None
    centroid = geometry.centroid
    return float(centroid.y), float(centroid.x)


def _geometry_type_or_none(geometry: Any) -> str | None:
    if geometry is None or geometry.is_empty:
        return None
    return geometry.geom_type


def normalize_osm_facilities(
    raw_gdf: gpd.GeoDataFrame,
    config: InfrastructureConfig,
    source_version: str,
) -> gpd.GeoDataFrame:
    """Build the canonical facility table from a raw loaded OSM extract.

    Args:
        raw_gdf: Output of `osm_loader.load_osm_extract` (columns
            ``osm_id``, ``osm_type``, ``name``, ``raw_tags``,
            ``geometry``).
        config: Pipeline configuration (source label, LNG keywords).
        source_version: Free-text provenance string recorded verbatim in
            every row's `source_version` (e.g. the input file name and
            modification time) and in the report.

    Returns:
        A `GeoDataFrame` with columns `facility_schema.CANONICAL_COLUMNS`
        plus a `geometry` column (the *original* geometry — polygons are
        never replaced by their centroid). One output row per input row;
        this function never drops or merges rows (see
        `facility_validation`/`infrastructure_pipeline` for that).
    """
    records: list[dict[str, Any]] = []
    geometries: list[Any] = []

    for row in raw_gdf.itertuples(index=False):
        tags: dict[str, Any] = dict(row.raw_tags) if isinstance(row.raw_tags, dict) else {}
        geometry = row.geometry

        name_value = row.name if isinstance(row.name, str) and row.name.strip() else None
        type_result = classify_facility_type(tags, name_value, config.lng_name_keywords)

        osm_id = str(row.osm_id).strip() if row.osm_id not in (None, "") and pd.notna(row.osm_id) else None
        osm_type = str(row.osm_type).strip().lower() if row.osm_type not in (None, "") and pd.notna(row.osm_type) else None

        tags_json = encode_tags(tags)
        lat, lon = _representative_point(geometry)

        facility_id = make_osm_facility_id(osm_type, osm_id)
        if facility_id is None:
            facility_id = make_fallback_facility_id(
                latitude=lat, longitude=lon, name=name_value, tags_json=tags_json
            )

        record = {
            "facility_id": facility_id,
            "facility_name": name_value,
            "facility_type": type_result.facility_type,
            "industrial_subtype": type_result.industrial_subtype,
            "operator": _raw_tag(tags, "operator"),
            "landuse": type_result.landuse,
            "power_type": type_result.power_type,
            "man_made_type": type_result.man_made_type,
            "confidence": type_result.confidence,
            "geometry_type": _geometry_type_or_none(geometry),
            "latitude": lat,
            "longitude": lon,
            "geometry_wkt": geometry.wkt if geometry is not None and not geometry.is_empty else None,
            "osm_id": osm_id,
            "osm_type": osm_type,
            "osm_tags": tags_json,
            "source": config.source_label,
            "source_version": source_version,
        }
        records.append(record)
        geometries.append(geometry)

    df = pd.DataFrame.from_records(records, columns=list(CANONICAL_COLUMNS))
    result = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return result


def facility_type_counts(df: pd.DataFrame) -> dict[str, int]:
    """Value counts of `facility_type`, including zero-count categories."""
    counts = df["facility_type"].value_counts()
    from src.infrastructure.facility_schema import FACILITY_TYPES

    return {t: int(counts.get(t, 0)) for t in FACILITY_TYPES}


def geometry_type_counts(df: pd.DataFrame) -> dict[str, int]:
    """Value counts of `geometry_type`, including zero-count supported types."""
    counts = df["geometry_type"].value_counts(dropna=False)
    result = {t: int(counts.get(t, 0)) for t in SUPPORTED_GEOMETRY_TYPES}
    other = int(len(df) - sum(result.values()))
    if other:
        result["OTHER_OR_MISSING"] = other
    return result
