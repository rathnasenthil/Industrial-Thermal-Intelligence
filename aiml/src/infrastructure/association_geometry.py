"""
Geometry handling and spatial-index candidate search for GIFT Stage I.2.

COORDINATE SYSTEM STRATEGY
------------------------------------------------------------------------
Facility geometry (`osm_facilities.geojson`, from Stage I.1) and thermal
event geometry (`centroid_wkt`/`footprint_wkt`, from Stage G) are both
persisted in WGS84 (EPSG:4326) -- ordinary latitude/longitude degrees.
Degrees are angular, not linear, so neither containment/intersection
testing nor distance measurement should be done directly on raw
lat/lon coordinates for anything spanning more than a trivial extent
(this project already avoids the classic
``sqrt((lat1-lat2)**2 + (lon1-lon2)**2)`` mistake).

This module reprojects both event and facility geometry into a single
custom **India-centered Albers Equal-Area Conic** projection before any
distance/buffer/containment operation:

    +proj=aea +lat_1=8 +lat_2=37 +lat_0=22 +lon_0=82
    +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs

Rationale for this specific choice (see `INDIA_EQUAL_AREA_CRS`):

* The standard parallels (8 deg N / 37 deg N) bracket mainland India's
  full latitude range, and the central meridian (82 deg E) sits near its
  longitudinal center -- a standard, textbook way to parameterize an
  Albers projection for a specific country/region, which keeps areal and
  (for this project's purposes) linear distortion low and roughly even
  across the whole study area, unlike a single UTM zone (which would
  need 6 zones to cover India without significant edge distortion) or
  unprojected degrees (where a degree of longitude is ~111 km at the
  equator but only ~85 km at 40 deg N).
* It is defined directly via a PROJ string rather than an EPSG code
  lookup, so it does not depend on a specific regional/national EPSG
  entry being present in the installed PROJ database -- it is fully
  self-contained and reproducible across environments.
* Distances and containment/intersection tests computed in this
  projected CRS are planar, not geodesic-exact -- adequate for this
  stage's engineering search-radius/candidate-ranking purpose (a few
  km of accuracy at India's scale), but NOT a survey-grade geodesic
  distance. This is consistent with -- and no less precise than -- the
  simple-mean "centroid" already used for event geometry in Stage G
  (`src.event_formation.geometry`), which likewise does not claim
  geodesic exactness.

Topological relationship tests (within/intersects) are computed on the
SAME projected geometries used for distance, so relation and distance
are always mutually consistent for a given candidate pair.

SPATIAL INDEX / CANDIDATE SEARCH STRATEGY
------------------------------------------------------------------------
This module never computes an events x facilities distance matrix (which
would be ~179,740 x 112,956 ~= 2*10^10 pairs -- clearly infeasible).
Instead it uses GeoPandas' spatial index (`GeoDataFrame.sjoin`, backed by
an STRtree) to find only the candidate pairs whose bounding geometry
actually overlaps:

    for each event: buffer its footprint geometry by association_radius_km
    -> spatial-index join against facility geometries (predicate="intersects")
    -> only surviving (event, facility) pairs get an exact distance/relation
       computation (fully vectorized via aligned GeoSeries operations, not
       a Python-level nested loop).

Buffering the event's *footprint* (its Stage G convex-hull spatial
extent) rather than only its centroid point is deliberately more
inclusive: because the centroid is always the arithmetic mean of the
member detections, it always lies within (or on) the footprint's convex
hull, so buffering the footprint can only find a superset of the
candidates that buffering the centroid point alone would find. The final
NEAR_FACILITY classification then re-checks the *exact* centroid-to-
facility distance against `association_radius_km` (see
`find_candidate_pairs`), so this wider net never lets in a facility
farther than the configured radius from the centroid without also being
directly touched by the event's own footprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

from src.infrastructure.facility_schema import SUPPORTED_GEOMETRY_TYPES

# See module docstring for the full rationale.
INDIA_EQUAL_AREA_CRS = (
    "+proj=aea +lat_1=8 +lat_2=37 +lat_0=22 +lon_0=82 "
    "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
)

WITHIN_FACILITY = "WITHIN_FACILITY"
INTERSECTS_FACILITY = "INTERSECTS_FACILITY"
NEAR_FACILITY = "NEAR_FACILITY"

FACILITY_COLUMNS: tuple[str, ...] = ("facility_id", "facility_name", "facility_type", "geometry_type")


class CandidatePairs(NamedTuple):
    """Raw candidate (event, facility) pairs with exact geometry results.

    Attributes:
        pairs_df: One row per candidate pair, columns: ``event_id``,
            ``facility_id``, ``facility_name``, ``facility_type``,
            ``geometry_type`` (facility's), ``distance_km`` (event
            centroid to facility geometry, planar in
            `INDIA_EQUAL_AREA_CRS`), ``spatial_relation`` (one of
            `WITHIN_FACILITY`/`INTERSECTS_FACILITY`/`NEAR_FACILITY`).
    """

    pairs_df: pd.DataFrame


def load_facilities_geodataframe(path: str | Path) -> gpd.GeoDataFrame:
    """Load the Stage I.1 normalized facility layer.

    Args:
        path: Path to `osm_facilities.geojson` (preferred) or
            `osm_facilities.csv` (uses its `geometry_wkt` column).

    Returns:
        A `GeoDataFrame` (EPSG:4326) with exactly `FACILITY_COLUMNS` plus
        `geometry`. Rows with missing/unsupported geometry are dropped
        here defensively (Stage I.1's own validation should already
        guarantee this for `osm_facilities.*`, but Stage I.2 must not
        crash if a hand-edited or older facility file slips through one
        anyway -- see edge case "missing/invalid geometry" in the test
        suite).

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If required columns are missing.
    """
    facility_path = Path(path)
    if not facility_path.exists():
        raise FileNotFoundError(f"Stage I.1 facility file not found: {facility_path}")

    suffix = facility_path.suffix.lower()
    if suffix in (".geojson", ".json"):
        gdf = gpd.read_file(facility_path)
    elif suffix == ".csv":
        df = pd.read_csv(facility_path)
        missing = [c for c in (*FACILITY_COLUMNS, "geometry_wkt") if c not in df.columns]
        if missing:
            raise ValueError(f"'{facility_path}' is missing required column(s): {missing}.")
        geometries = [
            shapely_wkt.loads(w) if isinstance(w, str) and w.strip() else None for w in df["geometry_wkt"]
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    else:
        raise ValueError(f"Unsupported facility file extension '{suffix}' for '{facility_path}'.")

    missing = [c for c in FACILITY_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(f"'{facility_path}' is missing required column(s): {missing}.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() not in {"EPSG:4326", "WGS84"}:
        gdf = gdf.to_crs("EPSG:4326")

    valid_geometry = (
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf["geometry_type"].isin(SUPPORTED_GEOMETRY_TYPES)
    )
    dropped = int((~valid_geometry).sum())
    if dropped:
        # Defensive only -- Stage I.1 validation should already exclude
        # these from osm_facilities.*. Never crash Stage I.2 over it;
        # such facilities simply cannot participate in spatial matching.
        gdf = gdf.loc[valid_geometry].reset_index(drop=True)

    return gdf[[*FACILITY_COLUMNS, "geometry"]].reset_index(drop=True)


REQUIRED_EVENT_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "event_id",
    "centroid_wkt",
    "footprint_wkt",
)


def build_event_geometries(events_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Parse Stage G event geometry columns into shapely geometries.

    Args:
        events_df: Thermal events table (Stage G / Stage G.1 output)
            with, at minimum, `REQUIRED_EVENT_GEOMETRY_COLUMNS`. Never
            modified.

    Returns:
        A `GeoDataFrame` (EPSG:4326) with columns ``event_id``,
        ``centroid_geom`` (always a `Point`), ``footprint_geom`` (the
        Stage G convex-hull footprint -- `Point`/`LineString`/`Polygon`
        depending on the event's detection count/collinearity; see
        `src.event_formation.geometry` module docstring for why this is
        an *observed-detection envelope*, not a physical fire boundary).
        The active `geometry` column is set to `centroid_geom` for
        convenience.

    Raises:
        ValueError: If required columns are missing, or a row's WKT
            cannot be parsed.
    """
    missing = [c for c in REQUIRED_EVENT_GEOMETRY_COLUMNS if c not in events_df.columns]
    if missing:
        raise ValueError(
            f"Events table is missing required column(s): {missing}. "
            "Stage I.2 expects the output of src.event_formation / "
            "src.persistence (thermal_events.csv or "
            "thermal_events_with_persistence.csv)."
        )

    try:
        centroid_geom = [shapely_wkt.loads(w) for w in events_df["centroid_wkt"]]
        footprint_geom = [shapely_wkt.loads(w) for w in events_df["footprint_wkt"]]
    except Exception as exc:  # noqa: BLE001 - re-raise with clearer context
        raise ValueError(f"Could not parse event centroid_wkt/footprint_wkt: {exc}") from exc

    gdf = gpd.GeoDataFrame(
        {
            "event_id": events_df["event_id"].to_numpy(),
            "centroid_geom": gpd.GeoSeries(centroid_geom, crs="EPSG:4326"),
            "footprint_geom": gpd.GeoSeries(footprint_geom, crs="EPSG:4326"),
        },
        geometry="centroid_geom",
        crs="EPSG:4326",
    )
    return gdf


def find_candidate_pairs(
    events_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    association_radius_km: float,
) -> pd.DataFrame:
    """Spatial-index candidate search + exact relation/distance computation.

    Never computes an events x facilities distance matrix; see module
    docstring for the buffer-and-join strategy and its complexity.

    Args:
        events_gdf: Output of `build_event_geometries` (EPSG:4326).
        facilities_gdf: Output of `load_facilities_geodataframe`
            (EPSG:4326).
        association_radius_km: See `AssociationConfig.association_radius_km`.

    Returns:
        A DataFrame with one row per retained (event, facility)
        candidate pair: ``event_id``, ``facility_id``, ``facility_name``,
        ``facility_type``, ``geometry_type``, ``distance_km``,
        ``spatial_relation``. Empty (but correctly shaped) if either
        input is empty or no candidates were found.
    """
    columns = [
        "event_id",
        "facility_id",
        "facility_name",
        "facility_type",
        "geometry_type",
        "distance_km",
        "spatial_relation",
    ]
    if len(events_gdf) == 0 or len(facilities_gdf) == 0:
        return pd.DataFrame(columns=columns)

    events_proj = events_gdf.copy()
    events_proj["centroid_proj"] = events_gdf["centroid_geom"].to_crs(INDIA_EQUAL_AREA_CRS)
    events_proj["footprint_proj"] = events_gdf["footprint_geom"].to_crs(INDIA_EQUAL_AREA_CRS)
    facilities_proj = facilities_gdf.copy()
    facilities_proj["geometry"] = facilities_gdf["geometry"].to_crs(INDIA_EQUAL_AREA_CRS)

    # Wide-net spatial-index search: buffer each event's footprint (which
    # always contains its own centroid -- an arithmetic mean always lies
    # within its points' convex hull) by the association radius, then
    # let the spatial index find every facility whose geometry overlaps
    # that buffer at all.
    buffered = gpd.GeoDataFrame(
        {"event_position": np.arange(len(events_proj))},
        geometry=events_proj["footprint_proj"].buffer(association_radius_km * 1000.0).values,
        crs=INDIA_EQUAL_AREA_CRS,
    )
    facility_search = facilities_proj[["geometry"]].copy()
    facility_search["facility_position"] = np.arange(len(facilities_proj))

    joined = gpd.sjoin(buffered, facility_search, how="inner", predicate="intersects")
    if joined.empty:
        return pd.DataFrame(columns=columns)

    left_positions = joined["event_position"].to_numpy()
    right_positions = joined["facility_position"].to_numpy()

    candidate_centroid = gpd.GeoSeries(
        events_proj["centroid_proj"].to_numpy()[left_positions], crs=INDIA_EQUAL_AREA_CRS
    ).reset_index(drop=True)
    candidate_footprint = gpd.GeoSeries(
        events_proj["footprint_proj"].to_numpy()[left_positions], crs=INDIA_EQUAL_AREA_CRS
    ).reset_index(drop=True)
    candidate_facility_geom = gpd.GeoSeries(
        facilities_proj["geometry"].to_numpy()[right_positions], crs=INDIA_EQUAL_AREA_CRS
    ).reset_index(drop=True)

    distance_km = candidate_centroid.distance(candidate_facility_geom) / 1000.0
    # Note: `within()` for a Point facility is topological point-equality,
    # so a Point-type facility can only ever be WITHIN_FACILITY in the
    # (rare, but not incorrect) edge case where the event centroid exactly
    # coincides with the facility's coordinates -- this is intentional,
    # since that is also exactly when distance_km == 0.
    within_mask = candidate_centroid.within(candidate_facility_geom).to_numpy()
    intersects_mask = candidate_footprint.intersects(candidate_facility_geom).to_numpy()
    distance_km_arr = distance_km.to_numpy()
    near_mask = (~within_mask) & (~intersects_mask) & (distance_km_arr <= association_radius_km)
    keep_mask = within_mask | intersects_mask | near_mask

    if not keep_mask.any():
        return pd.DataFrame(columns=columns)

    relation = np.where(
        within_mask, WITHIN_FACILITY, np.where(intersects_mask, INTERSECTS_FACILITY, NEAR_FACILITY)
    )

    event_ids = events_gdf["event_id"].to_numpy()[left_positions]
    facility_ids = facilities_gdf["facility_id"].to_numpy()[right_positions]
    facility_names = facilities_gdf["facility_name"].to_numpy()[right_positions]
    facility_types = facilities_gdf["facility_type"].to_numpy()[right_positions]
    geometry_types = facilities_gdf["geometry_type"].to_numpy()[right_positions]

    result = pd.DataFrame(
        {
            "event_id": event_ids[keep_mask],
            "facility_id": facility_ids[keep_mask],
            "facility_name": facility_names[keep_mask],
            "facility_type": facility_types[keep_mask],
            "geometry_type": geometry_types[keep_mask],
            "distance_km": distance_km_arr[keep_mask],
            "spatial_relation": relation[keep_mask],
        }
    )
    # distance_km must never be negative -- a planar/geodesic distance is
    # mathematically non-negative, but round to avoid float noise like
    # -0.0 from shapely for exact-boundary cases.
    result["distance_km"] = result["distance_km"].clip(lower=0.0)
    return result
