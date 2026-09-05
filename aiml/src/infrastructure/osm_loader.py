"""
Static OSM extract loading (GIFT Stage I.1).

This project does NOT depend on live Overpass API access, and this
module never downloads or fabricates OSM data. It only loads a static,
user-supplied extract file from disk. Two practical tabular formats are
supported:

* **GeoJSON** (``.geojson`` / ``.json``) — the typical output of
  ``overpass-turbo`` / ``osmtogeojson`` / QGIS "Export OSM" plugins: a
  ``FeatureCollection`` where each feature's ``properties`` holds the OSM
  tags (plus an id/type, in whatever convention the exporting tool used)
  and ``geometry`` is a Point/Polygon/MultiPolygon.
* **CSV** (``.csv``) — a flattened tabular export (e.g. from
  ``ogr2ogr``/QGIS), with either a ``geometry_wkt`` column (any geometry
  type) or plain ``latitude``/``longitude`` columns (points only), plus
  either a JSON-encoded ``tags`` column or arbitrary extra columns that
  are treated as individual OSM tags.

See ``aiml/README.md`` ("GIFT Stage I.1") for exactly where to place a
real extract and the expected column/property names. If no such file is
present, :func:`discover_default_osm_input` returns ``None`` — callers
(see ``infrastructure_pipeline.py``) must handle that explicitly rather
than pretending an extract exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt

# Property/column names that carry structural metadata rather than an OSM
# tag. Every other property/column in the input is treated as a raw OSM
# tag and preserved verbatim in `osm_tags`.
_RESERVED_GEOJSON_PROPERTIES: frozenset[str] = frozenset(
    {"osm_id", "osm_type", "id", "@id", "@type", "type", "name"}
)
_RESERVED_CSV_COLUMNS: frozenset[str] = frozenset(
    {"osm_id", "osm_type", "latitude", "longitude", "geometry_wkt", "tags", "name"}
)

# Patterns this project recognizes for a combined "<type>/<id>" identifier,
# as commonly produced by `osmtogeojson` (e.g. GeoJSON top-level `id` or a
# property such as `@id` with value ``"way/123456789"``).
_COMBINED_ID_PATTERN = re.compile(r"^(node|way|relation)[/:](\d+)$", re.IGNORECASE)

# Filename fragments this project looks for when auto-discovering a static
# extract in `aiml/data/raw/`. Matching is case-insensitive and deliberately
# broad (substring, not exact) since real-world export tools name files
# differently.
_DEFAULT_INPUT_NAME_HINTS: tuple[str, ...] = ("osm", "facility", "facilities", "industrial")
_SUPPORTED_SUFFIXES: tuple[str, ...] = (".geojson", ".json", ".csv", ".pbf")


class OsmInputError(ValueError):
    """Raised when a static OSM extract file cannot be parsed as expected."""


def _split_combined_id(value: Any) -> tuple[str | None, str | None]:
    """Parse a combined ``"way/123456789"``-style identifier.

    Returns:
        ``(osm_type, osm_id)``, both ``None`` if `value` doesn't match.
    """
    if value is None:
        return None, None
    match = _COMBINED_ID_PATTERN.match(str(value).strip())
    if not match:
        return None, None
    return match.group(1).lower(), match.group(2)


def _resolve_osm_id_type(properties: dict[str, Any], feature_id: Any) -> tuple[str | None, str | None]:
    """Best-effort extraction of ``(osm_type, osm_id)`` from a GeoJSON feature.

    Tries, in order: explicit ``osm_id``/``osm_type`` properties; a
    combined ``"<type>/<id>"`` value in the feature's top-level ``id`` or
    an ``@id``/``id`` property (the convention used by ``osmtogeojson``);
    otherwise gives up and returns ``(None, None)`` (the caller falls back
    to a deterministic content-based id — see `facility_schema`).
    """
    explicit_type = properties.get("osm_type")
    explicit_id = properties.get("osm_id")
    if not _is_missing(explicit_type) and not _is_missing(explicit_id):
        return str(explicit_type).strip().lower(), str(explicit_id).strip()

    for candidate in (feature_id, properties.get("@id"), properties.get("id")):
        osm_type, osm_id = _split_combined_id(candidate)
        if osm_type and osm_id:
            return osm_type, osm_id

    return None, None


def _is_missing(value: Any) -> bool:
    """True for None/NaN/empty-string values.

    IMPORTANT: `geopandas.read_file` builds one property column per *union*
    of all features' property keys, filling any feature that lacks a given
    key with float ``NaN`` (not ``None``). Without this explicit NaN check,
    those filled-in gaps would otherwise leak into `raw_tags` as a literal
    ``"nan"`` string (since ``str(float('nan')) == "nan"``, which is
    non-empty and therefore truthy) and corrupt tag-based classification.
    """
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _raw_tags_from_geojson_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Every property that isn't structural metadata is treated as an OSM tag."""
    return {
        str(k): v
        for k, v in properties.items()
        if k not in _RESERVED_GEOJSON_PROPERTIES and not _is_missing(v)
    }


def load_osm_geojson(path: str | Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON static OSM extract.

    Args:
        path: Path to a ``.geojson``/``.json`` FeatureCollection.

    Returns:
        A `GeoDataFrame` (CRS forced/assumed to be EPSG:4326 — the OSM/
        GeoJSON standard) with columns ``osm_id``, ``osm_type``, ``name``,
        ``raw_tags`` (a ``dict`` per row) and ``geometry``.

    Raises:
        FileNotFoundError: If `path` does not exist.
        OsmInputError: If the file cannot be read as a GeoDataFrame.
    """
    geojson_path = Path(path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"OSM GeoJSON extract not found: {geojson_path}")

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as exc:  # noqa: BLE001 - re-raise with clearer context
        raise OsmInputError(f"Could not parse '{geojson_path}' as GeoJSON: {exc}") from exc

    if gdf.crs is None:
        # GeoJSON's implicit CRS is WGS84 (EPSG:4326) per the spec; make it
        # explicit rather than silently relying on an unset CRS downstream.
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() not in {"EPSG:4326", "WGS84"}:
        gdf = gdf.to_crs("EPSG:4326")

    # geopandas exposes each feature's top-level GeoJSON `id` (if present)
    # via the frame index in some drivers; re-read raw features defensively
    # to also capture it when available.
    feature_ids: list[Any] = [None] * len(gdf)
    try:
        with geojson_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        features = raw.get("features", []) if isinstance(raw, dict) else []
        for i, feature in enumerate(features[: len(gdf)]):
            feature_ids[i] = feature.get("id")
    except (OSError, ValueError, json.JSONDecodeError):
        pass  # feature-level id is best-effort only; property-based id still works.

    property_columns = [c for c in gdf.columns if c != "geometry"]
    osm_ids: list[str | None] = []
    osm_types: list[str | None] = []
    names: list[str | None] = []
    raw_tags: list[dict[str, Any]] = []

    for i, row in gdf[property_columns].iterrows():
        properties = {c: row[c] for c in property_columns}
        osm_type, osm_id = _resolve_osm_id_type(properties, feature_ids[i] if i < len(feature_ids) else None)
        osm_ids.append(osm_id)
        osm_types.append(osm_type)
        name_value = properties.get("name")
        names.append(str(name_value).strip() if name_value not in (None, "") else None)
        raw_tags.append(_raw_tags_from_geojson_properties(properties))

    result = gpd.GeoDataFrame(
        {
            "osm_id": osm_ids,
            "osm_type": osm_types,
            "name": names,
            "raw_tags": raw_tags,
            "geometry": gdf.geometry.values,
        },
        crs="EPSG:4326",
    )
    return result


def load_osm_csv(path: str | Path) -> gpd.GeoDataFrame:
    """Load a flattened CSV static OSM extract.

    Expected columns (all optional except a geometry source):

    * ``osm_id``, ``osm_type`` — stable OSM identifier, if available.
    * ``name`` — facility name, if available.
    * ``geometry_wkt`` — a WKT geometry string (any type); preferred when
      present since it supports polygons/multipolygons, not just points.
    * ``latitude``, ``longitude`` — used only if ``geometry_wkt`` is
      absent (points only).
    * ``tags`` — a JSON-encoded dict of OSM tags. If absent, every other,
      non-reserved column in the CSV is treated as an individual tag.

    Args:
        path: Path to a ``.csv`` file.

    Returns:
        A `GeoDataFrame` with the same columns as :func:`load_osm_geojson`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        OsmInputError: If no usable geometry source column is present, or
            a row's geometry cannot be parsed.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OSM CSV extract not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    has_wkt = "geometry_wkt" in df.columns
    has_latlon = "latitude" in df.columns and "longitude" in df.columns
    if not has_wkt and not has_latlon:
        raise OsmInputError(
            f"'{csv_path}' has neither a 'geometry_wkt' column nor "
            "'latitude'/'longitude' columns — cannot build geometry. "
            "See aiml/README.md (GIFT Stage I.1) for the expected CSV schema."
        )

    geometries = []
    for i, row in df.iterrows():
        if has_wkt and pd.notna(row.get("geometry_wkt")):
            try:
                geometries.append(shapely_wkt.loads(row["geometry_wkt"]))
            except Exception as exc:  # noqa: BLE001
                raise OsmInputError(
                    f"'{csv_path}' row {i}: could not parse geometry_wkt value "
                    f"{row['geometry_wkt']!r}: {exc}"
                ) from exc
        elif has_latlon and pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            from shapely.geometry import Point

            try:
                geometries.append(Point(float(row["longitude"]), float(row["latitude"])))
            except (TypeError, ValueError) as exc:
                raise OsmInputError(f"'{csv_path}' row {i}: invalid latitude/longitude: {exc}") from exc
        else:
            geometries.append(None)  # no usable geometry for this row; flagged later in validation.

    tag_columns = [c for c in df.columns if c not in _RESERVED_CSV_COLUMNS]

    osm_ids = df["osm_id"].tolist() if "osm_id" in df.columns else [None] * len(df)
    osm_types = df["osm_type"].tolist() if "osm_type" in df.columns else [None] * len(df)
    names = df["name"].tolist() if "name" in df.columns else [None] * len(df)

    raw_tags: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        if "tags" in df.columns and pd.notna(row.get("tags")):
            try:
                parsed = json.loads(row["tags"])
                tags = dict(parsed) if isinstance(parsed, dict) else {}
            except (TypeError, ValueError):
                tags = {}
        else:
            tags = {c: row[c] for c in tag_columns if pd.notna(row.get(c))}
        raw_tags.append(tags)

    result = gpd.GeoDataFrame(
        {
            "osm_id": osm_ids,
            "osm_type": osm_types,
            "name": names,
            "raw_tags": raw_tags,
            "geometry": geometries,
        },
        crs="EPSG:4326",
    )
    return result


def load_osm_extract(path: str | Path) -> gpd.GeoDataFrame:
    """Load a static OSM extract, dispatching on file extension.

    Args:
        path: Path to a ``.geojson``/``.json``, ``.csv``, or ``.pbf`` file.

    Returns:
        A `GeoDataFrame` with columns ``osm_id``, ``osm_type``, ``name``,
        ``raw_tags``, ``geometry`` (CRS EPSG:4326).

    Raises:
        FileNotFoundError: If `path` does not exist.
        OsmInputError: If the extension is unsupported or the file cannot
            be parsed.

    Note:
        For ``.pbf`` inputs, this convenience wrapper discards the
        streaming scan statistics (`osm_pbf_loader.PbfScanStats`). Callers
        that need those for reporting (e.g. `infrastructure_pipeline`)
        should call `osm_pbf_loader.load_osm_pbf` directly instead.
    """
    extract_path = Path(path)
    suffix = extract_path.suffix.lower()
    if suffix in (".geojson", ".json"):
        return load_osm_geojson(extract_path)
    if suffix == ".csv":
        return load_osm_csv(extract_path)
    if suffix == ".pbf":
        from src.infrastructure.osm_pbf_loader import load_osm_pbf

        raw_gdf, _stats = load_osm_pbf(extract_path)
        return raw_gdf
    raise OsmInputError(
        f"Unsupported OSM extract file extension '{suffix}' for '{extract_path}'. "
        f"Supported extensions: {_SUPPORTED_SUFFIXES}."
    )


def discover_default_osm_input(raw_dir: str | Path) -> Path | None:
    """Look for a user-supplied static OSM extract in `raw_dir`.

    This function NEVER downloads or fabricates a file — it only looks
    for one that already exists on disk. Matching is deliberately broad
    (case-insensitive substring on the filename) since real-world export
    tools name files differently; see `aiml/README.md` for the exact
    recommended filename.

    Args:
        raw_dir: Directory to search (typically ``aiml/data/raw``).

    Returns:
        The first matching file path (sorted alphabetically for
        determinism), or ``None`` if no candidate file is found (this is
        the expected outcome until a real extract is placed there).
    """
    directory = Path(raw_dir)
    if not directory.exists():
        return None

    candidates = sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in _SUPPORTED_SUFFIXES
        and any(hint in p.name.lower() for hint in _DEFAULT_INPUT_NAME_HINTS)
    )
    return candidates[0] if candidates else None
