"""
End-to-end OSM Facility Ingestion & Normalization pipeline (GIFT Stage I.1).

Wires together, in order:

1. Load a static OSM extract (`osm_loader.load_osm_extract`) -- or, if
   none is available, explicitly record that fact rather than
   fabricating one (`run_infrastructure_ingestion(input_path=None, ...)`).
2. Normalize raw OSM tags into the canonical facility schema
   (`osm_normalization.normalize_osm_facilities`).
3. Validate every record and detect duplicate facility ids
   (`facility_validation`).
4. Assemble a reproducibility/statistics report
   (`facility_report.build_facility_report`).

This stage NEVER reads or modifies `thermal_events.csv` or
`thermal_events_with_persistence.csv` -- there is no thermal-event
association here (that is Stage I.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.infrastructure.config import InfrastructureConfig
from src.infrastructure.facility_report import (
    LOADED_STATUS,
    NO_PRODUCTION_INPUT_STATUS,
    PRODUCTION_PBF_STATUS,
    build_facility_report,
)
from src.infrastructure.facility_schema import CANONICAL_COLUMNS
from src.infrastructure.facility_validation import (
    build_rejection_reasons,
    detect_duplicate_facility_ids,
    validate_facilities,
)
from src.infrastructure.osm_loader import load_osm_extract
from src.infrastructure.osm_normalization import normalize_osm_facilities
from src.infrastructure.osm_pbf_loader import load_osm_pbf


@dataclass
class InfrastructureResult:
    """Result of running the Stage I.1 pipeline.

    Attributes:
        facilities_gdf: Final, valid, de-duplicated canonical facility
            table (one row per facility). Empty (but correctly shaped)
            when no production OSM input was available.
        rejected_df: Records excluded from `facilities_gdf` (invalid
            geometry/coordinates/id/type, or duplicate `facility_id`),
            each with a `rejection_reason` -- preserved, never deleted.
        report: JSON-serializable Stage I.1 report.
    """

    facilities_gdf: gpd.GeoDataFrame
    rejected_df: pd.DataFrame
    report: dict[str, Any]


def _empty_canonical_gdf() -> gpd.GeoDataFrame:
    columns = {c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS}
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], dtype="geometry"), crs="EPSG:4326")


def describe_source_version(path: Path) -> str:
    """Default `source_version` provenance string for a static extract.

    Combines the file name and last-modified time so the report always
    records *which* extract (and when it was captured/exported) produced
    a given facility layer, without requiring the user to pass anything
    explicitly.
    """
    mtime = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
    return f"{path.name} (mtime={mtime})"


def run_infrastructure_ingestion(
    input_path: str | Path | None,
    config: InfrastructureConfig,
    source_version: str | None = None,
) -> InfrastructureResult:
    """Run the full Stage I.1 pipeline.

    Args:
        input_path: Path to a static OSM extract (GeoJSON/CSV), or
            ``None`` if no production input is available. When ``None``,
            this function does NOT fail or fabricate data -- it returns
            an explicitly empty result whose report clearly states no
            coverage exists (see `facility_report.NO_PRODUCTION_INPUT_STATUS`).
        config: Pipeline configuration.
        source_version: Provenance string for the report/output rows. If
            ``None`` and `input_path` is given, defaults to
            :func:`describe_source_version`.

    Returns:
        An `InfrastructureResult`.
    """
    start_time = time.perf_counter()

    if input_path is None:
        empty = _empty_canonical_gdf()
        report = build_facility_report(
            config=config,
            input_status=NO_PRODUCTION_INPUT_STATUS,
            source_path=None,
            source_version=None,
            raw_record_count=0,
            normalized_df=empty,
            validation_stats=validate_facilities(empty).stats,
            duplicate_stats=detect_duplicate_facility_ids(empty).stats,
            final_df=empty,
            rejected_df=pd.DataFrame(),
            processing_seconds=time.perf_counter() - start_time,
        )
        return InfrastructureResult(facilities_gdf=empty, rejected_df=pd.DataFrame(), report=report)

    resolved_path = Path(input_path)
    if source_version is None:
        source_version = describe_source_version(resolved_path)

    is_pbf = resolved_path.suffix.lower() == ".pbf"
    pbf_scan_stats: dict[str, Any] | None = None
    if is_pbf:
        # Streaming PBF path: `load_osm_pbf` never materializes the whole
        # country in memory -- only tag-filtered candidates are returned.
        # See `osm_pbf_loader` module docstring for the full rationale.
        raw_gdf, scan_stats = load_osm_pbf(resolved_path)
        pbf_scan_stats = scan_stats.to_dict()
        input_status = PRODUCTION_PBF_STATUS
    else:
        raw_gdf = load_osm_extract(resolved_path)
        input_status = LOADED_STATUS

    normalized_gdf = normalize_osm_facilities(raw_gdf, config, source_version)

    validation = validate_facilities(normalized_gdf)
    duplicates = detect_duplicate_facility_ids(normalized_gdf)

    keep_mask = validation.valid_mask & (~duplicates.duplicate_mask)
    final_gdf = normalized_gdf.loc[keep_mask].reset_index(drop=True)

    rejected_mask = ~keep_mask
    rejected_df = pd.DataFrame(normalized_gdf.loc[rejected_mask]).reset_index(drop=True)
    if not rejected_df.empty:
        rejected_reasons = build_rejection_reasons(normalized_gdf, validation, duplicates.duplicate_mask)
        rejected_df["rejection_reason"] = rejected_reasons.loc[rejected_mask].reset_index(drop=True)

    processing_seconds = time.perf_counter() - start_time

    report = build_facility_report(
        config=config,
        input_status=input_status,
        source_path=str(resolved_path),
        source_version=source_version,
        raw_record_count=len(raw_gdf),
        normalized_df=normalized_gdf,
        validation_stats=validation.stats,
        duplicate_stats=duplicates.stats,
        final_df=final_gdf,
        rejected_df=rejected_df,
        processing_seconds=processing_seconds,
        file_size_bytes=resolved_path.stat().st_size,
        pbf_scan_stats=pbf_scan_stats,
    )

    return InfrastructureResult(facilities_gdf=final_gdf, rejected_df=rejected_df, report=report)


def save_outputs(
    result: InfrastructureResult,
    facilities_csv_path: str | Path,
    facilities_geojson_path: str | Path,
    rejected_csv_path: str | Path | None = None,
) -> None:
    """Write the final facility table to CSV and GeoJSON (and, optionally, rejects to CSV).

    Args:
        result: Output of `run_infrastructure_ingestion`.
        facilities_csv_path: Destination for `osm_facilities.csv` (all
            canonical columns; the `geometry` object column is dropped,
            `geometry_wkt` retains the geometry as text).
        facilities_geojson_path: Destination for `osm_facilities.geojson`
            (real geometry, plus all canonical columns as properties).
        rejected_csv_path: Optional destination for rejected records
            (with `rejection_reason`). Skipped if ``None`` or if there are
            no rejected records.
    """
    csv_path = Path(facilities_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.facilities_gdf.drop(columns=["geometry"], errors="ignore").to_csv(csv_path, index=False)

    geojson_path = Path(facilities_geojson_path)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    if geojson_path.exists():
        geojson_path.unlink()  # to_file(driver="GeoJSON") can error if an old file with different schema exists.
    result.facilities_gdf.to_file(geojson_path, driver="GeoJSON")

    if rejected_csv_path is not None and not result.rejected_df.empty:
        rejected_path = Path(rejected_csv_path)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        result.rejected_df.drop(columns=["geometry"], errors="ignore").to_csv(rejected_path, index=False)
