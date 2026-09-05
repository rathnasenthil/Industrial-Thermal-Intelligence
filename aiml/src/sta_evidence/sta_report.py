"""JSON report assembly for GIFT Stage I.5."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS
from src.sta_evidence.config import STAConfig


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_sta_report(
    *,
    config: STAConfig,
    events_input_path: str,
    output_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    load_stats: dict[str, Any],
    validation_stats: dict[str, Any],
    processing_seconds: float,
    status: str = "production_sta_integrated",
) -> dict[str, Any]:
    status_counts = output_df["sta_association_status"].value_counts()
    quality_counts = output_df["sta_evidence_quality"].value_counts()
    layer_counts = output_df.loc[output_df["sta_layer_type"].notna(), "sta_layer_type"].value_counts()
    temporal_counts = output_df.loc[output_df["sta_temporal_relation"].notna(), "sta_temporal_relation"].value_counts()

    distances = output_df.loc[output_df["sta_nearest_distance_km"].notna(), "sta_nearest_distance_km"]
    distance_stats = (
        {
            "min": float(distances.min()),
            "median": float(distances.median()),
            "mean": round(float(distances.mean()), 4),
            "max": float(distances.max()),
            "count": int(len(distances)),
        }
        if not distances.empty
        else {"min": None, "median": None, "mean": None, "max": None, "count": 0}
    )

    match_counts = output_df["sta_match_count"]
    candidate_stats = {
        "events_with_zero_candidates": int((match_counts == 0).sum()),
        "events_with_one_candidate": int((match_counts == 1).sum()),
        "events_with_multiple_candidates": int((match_counts > 1).sum()),
        "max_candidates_per_event": int(match_counts.max()) if len(match_counts) else 0,
        "candidate_rows": int(len(candidates_df)),
    }

    # Bounding box from valid associations' distances not enough — use config provenance.
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": "GIFT Stage I.5 - NASA Static Thermal Anomaly Evidence Integration",
        "status": status,
        "input": {
            "events_input_path": events_input_path,
            "event_count": int(len(output_df)),
            "sta_records_read": validation_stats.get("records_read"),
            "sta_records_valid": validation_stats.get("records_valid"),
            "sta_source": config.sta_source,
            "sta_source_version": config.sta_source_version,
            "sta_source_url": config.sta_source_url,
            "sta_documentation_url": config.sta_documentation_url,
            "sta_download_date": config.sta_download_date,
            "storage_crs": "EPSG:4326",
            "computation_crs": INDIA_EQUAL_AREA_CRS,
            "load_stats": load_stats,
        },
        "validation": validation_stats,
        "spatial_matching": {
            "STA_ASSOCIATED": int(status_counts.get("STA_ASSOCIATED", 0)),
            "AMBIGUOUS": int(status_counts.get("AMBIGUOUS", 0)),
            "NO_STA_ASSOCIATION": int(status_counts.get("NO_STA_ASSOCIATION", 0)),
            "relationship_among_candidates": (
                candidates_df["relationship"].value_counts().to_dict() if not candidates_df.empty else {}
            ),
        },
        "layer_type_among_primary_associations": {str(k): int(v) for k, v in layer_counts.items()},
        "evidence_quality_counts": {
            "NONE": int(quality_counts.get("NONE", 0)),
            "LOW": int(quality_counts.get("LOW", 0)),
            "MEDIUM": int(quality_counts.get("MEDIUM", 0)),
            "HIGH": int(quality_counts.get("HIGH", 0)),
        },
        "distance_statistics_km": distance_stats,
        "candidate_statistics": candidate_stats,
        "temporal_statistics": {str(k): int(v) for k, v in temporal_counts.items()},
        "configuration": {
            **config.to_dict(),
            "rationale": config.describe_rationale(),
        },
        "i4_immutability": {
            "anomaly_fields_unchanged": True,
            "notes": (
                "I.5 appends STA evidence columns only. anomaly_score, anomaly_status, "
                "anomaly_confidence and feature deviations from Stage I.4 are never recalculated."
            ),
        },
        "performance": {"processing_seconds": round(processing_seconds, 3)},
        "reproducibility": {
            "deterministic": True,
            "notes": (
                "Candidate ranking uses a fixed sort key (relationship tier, distance, "
                "intersection area, layer priority, sta_id) with mergesort. Output events "
                "are sorted by event_id."
            ),
        },
        "limitations": [
            "NASA FIRMS STA Mask and STA Detections are experimental/provisional layers "
            "and may evolve; they are not authoritative ground truth.",
            "Industrial/natural heat-source inventories used to filter the STA Mask may "
            "be incomplete or inaccurate (NASA FIRMS disclaimer).",
            "STA Mask represents persistent/static thermal activity over its source "
            "construction period (e.g. calendar-year aggregation described by NASA) and "
            "must not be interpreted as a timestamped fire event.",
            "STA DETECTION timestamps, when present, are used conservatively; MASK "
            "features always have temporal relation NOT_APPLICABLE.",
            "Spatial association uses an engineering radius "
            f"({config.association_radius_km} km) and Stage G detection envelopes — not "
            "physical fire perimeters.",
            "NO_STA_ASSOCIATION does NOT imply the event is non-industrial.",
            "STA presence does NOT establish that a thermal event is an industrial fire.",
            "STA evidence and OSM/facility association (I.2) may share spatial biases and "
            "must not be treated as independent mutual validation.",
            "No source classification, ML, risk scoring, or I.4 anomaly recalculation is "
            "performed in this stage.",
            "No undocumented NASA bulk-download endpoint is hard-coded; local STA files "
            "must be supplied under data/raw/.",
        ],
    }
    return _to_jsonable(report)


def build_missing_source_report(config: STAConfig, events_count: int | None = None) -> dict[str, Any]:
    """Report when production STA files are absent — no fabricated data."""
    return _to_jsonable(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_stage": "GIFT Stage I.5 - NASA Static Thermal Anomaly Evidence Integration",
            "status": "production_sta_source_missing",
            "input": {
                "event_count": events_count,
                "mask_path": str(config.mask_path) if config.mask_path else None,
                "detection_path": str(config.detection_path) if config.detection_path else None,
                "sta_source": config.sta_source,
                "sta_source_url": config.sta_source_url,
                "sta_documentation_url": config.sta_documentation_url,
            },
            "message": (
                "No local NASA FIRMS STA Mask/Detections file was found. "
                "I.5 did not fabricate STA geometries. Place downloaded STA extracts "
                "under aiml/data/raw/ and re-run."
            ),
            "configuration": config.to_dict(),
            "limitations": [
                "Production STA integration requires a locally supplied NASA STA vector extract.",
                "STA layers are experimental/provisional and are not ground truth.",
            ],
        }
    )


def save_report(report: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
