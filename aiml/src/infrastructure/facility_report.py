"""
Report assembly for GIFT Stage I.1 (OSM Facility Ingestion & Normalization).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.infrastructure.config import InfrastructureConfig
from src.infrastructure.osm_normalization import facility_type_counts, geometry_type_counts

NO_PRODUCTION_INPUT_STATUS = "no_production_osm_input_found"
LOADED_STATUS = "static_extract_loaded"
PRODUCTION_PBF_STATUS = "production_osm_pbf"


def build_facility_report(
    *,
    config: InfrastructureConfig,
    input_status: str,
    source_path: str | None,
    source_version: str | None,
    raw_record_count: int,
    normalized_df: pd.DataFrame,
    validation_stats: dict[str, int],
    duplicate_stats: dict[str, int],
    final_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    processing_seconds: float,
    file_size_bytes: int | None = None,
    pbf_scan_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the Stage I.1 report as a JSON-serializable dict.

    Args:
        config: The `InfrastructureConfig` used for this run.
        input_status: `NO_PRODUCTION_INPUT_STATUS` if no static OSM
            extract was found/supplied, `PRODUCTION_PBF_STATUS` for a real
            OSM PBF input, else `LOADED_STATUS` (GeoJSON/CSV extract).
        source_path: Path to the OSM extract actually used, or ``None``.
        source_version: Provenance string recorded on every facility
            row, or ``None`` when there was no input.
        raw_record_count: Number of rows in the raw loaded extract,
            before normalization (should equal `len(normalized_df)` --
            normalization never drops rows). For PBF input, this is the
            number of *candidate* objects that passed the early tag
            filter -- see `pbf_scan_stats` for the full country-wide scan
            counts.
        normalized_df: Full canonical facility table, before
            validation/dedup filtering.
        validation_stats: Output of `facility_validation.validate_facilities`.
        duplicate_stats: Output of `facility_validation.detect_duplicate_facility_ids`.
        final_df: The final, valid, de-duplicated facility table that was
            (or will be) written to `osm_facilities.csv`/`.geojson`.
        rejected_df: Rows excluded from `final_df` (invalid and/or
            duplicate), preserved with a `rejection_reason` column -- never
            silently deleted.
        processing_seconds: Wall-clock seconds spent on normalization +
            validation.
        file_size_bytes: Size of the input file in bytes, if known.
        pbf_scan_stats: `osm_pbf_loader.PbfScanStats.to_dict()` output
            when `input_status` is `PRODUCTION_PBF_STATUS`, else ``None``.
            Includes the full country-wide `osm_objects_scanned` count
            (nodes + ways + relations seen, not just candidates).

    Returns:
        A JSON-serializable dict.
    """
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": (
            "GIFT Stage I.1 - Infrastructure & Environmental Context: "
            "OSM Facility Ingestion & Normalization"
        ),
        "input": {
            "status": input_status,
            "source_path": source_path,
            "file_size_bytes": int(file_size_bytes) if file_size_bytes is not None else None,
            "source_label": config.source_label,
            "source_version": source_version,
            "raw_record_count": int(raw_record_count),
            "pbf_scan_stats": pbf_scan_stats,
        },
        "normalization": {
            "normalized_record_count": int(len(normalized_df)),
            "facility_type_counts": (
                facility_type_counts(normalized_df) if len(normalized_df) else {}
            ),
            "geometry_type_counts": (
                geometry_type_counts(normalized_df) if len(normalized_df) else {}
            ),
        },
        "validation": validation_stats,
        "duplicate_detection": duplicate_stats,
        "output": {
            "final_facility_count": int(len(final_df)),
            "rejected_record_count": int(len(rejected_df)),
            "final_facility_type_counts": (facility_type_counts(final_df) if len(final_df) else {}),
            "final_geometry_type_counts": (geometry_type_counts(final_df) if len(final_df) else {}),
        },
        "coverage_status": (
            "NO REAL OSM COVERAGE: no production static OSM extract was found; "
            "outputs are empty and do NOT represent real-world facility "
            "coverage. See 'notes' below for where to place a real extract."
            if input_status == NO_PRODUCTION_INPUT_STATUS
            else (
                "A real, production OSM PBF extract was streamed and normalized "
                "(see 'input.pbf_scan_stats' for country-wide scan counts). "
                "Coverage is exactly whatever OSM mappers have tagged in that "
                "extract -- this stage makes no claim about completeness of "
                "real-world OSM coverage for the study area."
                if input_status == PRODUCTION_PBF_STATUS
                else (
                    "A static OSM extract was loaded and normalized. Coverage is "
                    "exactly whatever that input file contains -- this stage makes "
                    "no claim about completeness of real-world OSM coverage for "
                    "the study area."
                )
            )
        ),
        "performance": {"processing_seconds": round(processing_seconds, 3)},
        "reproducibility": {
            "deterministic": True,
            "notes": (
                "Given the same input extract file, facility_id values, "
                "facility_type classifications and all statistics above are "
                "identical across repeated runs. No random sampling or "
                "network access is used."
            ),
        },
        "notes": [
            "OSM is used here as CONTEXTUAL evidence only -- it is NOT ground "
            "truth for whether a nearby thermal event is an industrial fire. "
            "A facility record existing (or not existing) near a thermal "
            "event says nothing on its own about that event's cause.",
            "Missing OSM coverage for a real facility (a common real-world "
            "OSM data-completeness gap, especially in parts of India) must "
            "NOT be used later to automatically rule out an industrial cause "
            "for a thermal event -- absence of OSM data is not evidence of "
            "absence of infrastructure.",
            "UNKNOWN and OTHER_INDUSTRIAL are legitimate, expected "
            "facility_type outcomes for OSM objects that cannot be "
            "confidently mapped to a specific category -- not data-quality "
            "errors.",
            "This stage performs NO thermal-event association: "
            "distance_to_facility, facility_association, "
            "is_within_facility_boundary and attribution_confidence are "
            "computed later, in Stage I.2. thermal_events.csv and "
            "thermal_events_with_persistence.csv are never read or modified "
            "by this stage.",
            "Invalid and duplicate records are preserved (never silently "
            "deleted); see the rejected-records output and "
            "'rejection_reason' values.",
            f"To provide real coverage, place a static OSM extract (GeoJSON "
            f"or CSV) in '{config.raw_dir}' -- see aiml/README.md (GIFT Stage "
            f"I.1) for the exact expected filename hints and schema. This "
            f"pipeline never queries the live Overpass API and never "
            f"fabricates facility records.",
        ]
        + (
            [
                "PBF INPUT LIMITATION: full multipolygon RELATION geometry "
                "reconstruction is not implemented for this input format. "
                "Candidate relations (those with relevant industrial/power/"
                "mining tags) are preserved with their osm_id/osm_tags but "
                "geometry_type=None, so validation correctly flags and "
                "preserves them (with a rejection_reason) rather than "
                "silently dropping them -- see "
                "'validation.invalid_geometry_count' and the rejected-records "
                "output. Simple closed-way facility boundaries (the majority "
                "of real-world OSM industrial/power/mining polygons) ARE "
                "fully supported.",
            ]
            if input_status == PRODUCTION_PBF_STATUS
            else []
        ),
    }
    return _to_jsonable(report)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write the report dict to disk as pretty-printed JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
