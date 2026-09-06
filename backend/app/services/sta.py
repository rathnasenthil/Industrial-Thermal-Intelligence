"""
Backend Phase 8: incremental Stage I.5 NASA STA evidence.

Loads one ThermalEvent geometry context, calls AIML ``realtime.sta``,
and writes I.5 columns onto **that event only**.

STA is supporting evidence — not ground truth / industrial-fire
classification / risk scoring.

Does **not** call ``run_sta_integration()`` over all historical events.
Does **not** fabricate STA geometries when local NASA files are absent.
Does **not** modify I.3 fingerprint tables or I.4 anomaly fields.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.thermal_event import ThermalEvent
from app.services.facility_association import ensure_event_geometry_wkt

logger = logging.getLogger(__name__)

_AIML_ROOT = Path(__file__).resolve().parents[3] / "aiml"
if str(_AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(_AIML_ROOT))

from realtime.sta import (  # noqa: E402
    STAEvidenceResult,
    process_event_sta,
    unavailable_sta_result,
)
from src.sta_evidence.config import DEFAULT_CONFIG, STAConfig  # noqa: E402
from src.sta_evidence.sta_loader import (  # noqa: E402
    resolve_existing_paths,
    load_all_sta_layers,
)

# Lazy cache: load local STA vectors once per process when files exist.
_STA_LAYER_CACHE: dict[str, Any] | None = None


@dataclass
class STAEvidenceStats:
    events_updated: int = 0
    event_ids: list[str] = field(default_factory=list)
    by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _aiml_sta_config(config: Optional[STAConfig] = None) -> STAConfig:
    """Resolve STA paths relative to the aiml/ package root."""
    base = config or DEFAULT_CONFIG
    mask = base.mask_path
    det = base.detection_path
    if mask is not None and not Path(mask).is_absolute():
        mask = _AIML_ROOT / mask
    if det is not None and not Path(det).is_absolute():
        det = _AIML_ROOT / det
    return STAConfig(
        association_radius_km=base.association_radius_km,
        ambiguity_distance_tolerance_km=base.ambiguity_distance_tolerance_km,
        near_event_time_hours=base.near_event_time_hours,
        max_candidates_per_event=base.max_candidates_per_event,
        layer_priority=dict(base.layer_priority),
        mask_path=mask,
        detection_path=det,
        events_path=base.events_path,
        sta_source=base.sta_source,
        sta_source_url=base.sta_source_url,
        sta_documentation_url=base.sta_documentation_url,
        sta_source_version=base.sta_source_version,
        sta_download_date=base.sta_download_date,
    )


def get_cached_sta_gdf(config: STAConfig):
    """
    Return (sta_gdf, load_stats) or (None, None) when sources are missing.

    Does not fabricate data. Cache key is the resolved existing paths.
    """
    global _STA_LAYER_CACHE
    pairs = resolve_existing_paths(config)
    if not pairs:
        return None, None
    key = tuple((str(p), layer) for p, layer in pairs)
    if _STA_LAYER_CACHE is not None and _STA_LAYER_CACHE.get("key") == key:
        return _STA_LAYER_CACHE["gdf"], _STA_LAYER_CACHE["stats"]
    gdf, stats = load_all_sta_layers(config)
    _STA_LAYER_CACHE = {"key": key, "gdf": gdf, "stats": stats}
    return gdf, stats


def clear_sta_layer_cache() -> None:
    """Test helper: drop the process-local STA cache."""
    global _STA_LAYER_CACHE
    _STA_LAYER_CACHE = None


def _event_to_row(event: ThermalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_start": event.event_start,
        "event_end": event.event_end,
        "footprint_wkt": event.footprint_wkt,
        "centroid_wkt": event.centroid_wkt,
        "centroid_latitude": event.centroid_latitude,
        "centroid_longitude": event.centroid_longitude,
        # I.4 fields present so batch pipeline invariants can compare if needed;
        # realtime adapter never recalculates them.
        "anomaly_score": event.anomaly_score,
        "anomaly_status": event.anomaly_status,
        "anomaly_confidence": event.anomaly_confidence,
        "peak_frp_deviation": event.peak_frp_deviation,
        "event_size_deviation": event.event_size_deviation,
        "duration_deviation": event.duration_deviation,
        "distance_deviation": event.distance_deviation,
        "persistence_deviation": event.persistence_deviation,
        "monthly_deviation": event.monthly_deviation,
    }


def apply_sta_result(event: ThermalEvent, result: STAEvidenceResult) -> None:
    """Write I.5 columns onto one ThermalEvent (does not touch I.3/I.4)."""
    event.sta_association_status = result.sta_association_status
    event.primary_sta_id = result.primary_sta_id
    event.sta_layer_type = result.sta_layer_type
    event.sta_match_count = result.sta_match_count
    event.sta_nearest_distance_km = result.sta_nearest_distance_km
    event.sta_intersection_area_m2 = result.sta_intersection_area_m2
    event.sta_evidence_available = result.sta_evidence_available
    event.sta_temporal_relation = result.sta_temporal_relation
    event.sta_evidence_quality = result.sta_evidence_quality


def refresh_event_sta(
    session: Session,
    event_id: str,
    *,
    config: Optional[STAConfig] = None,
    sta_gdf=None,
) -> STAEvidenceResult:
    """
    Phase 8 entry: recompute I.5 for one event after Phase 7.

    When local NASA STA files are absent, writes NO_STA_ASSOCIATION defaults
    (same as zero candidates) and sets ``source_missing`` on the result.
    """
    cfg = _aiml_sta_config(config)
    event = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        raise RuntimeError(f"Event missing for STA refresh: {event_id}")

    # Ensure Stage G geometry WKT (Phase 5 may already have set this).
    ensure_event_geometry_wkt(session, event)
    session.refresh(event)

    if event.footprint_wkt is None and event.centroid_wkt is None:
        result = unavailable_sta_result(event_id, source_missing=False)
        apply_sta_result(event, result)
        session.flush()
        return result

    events_df = pd.DataFrame([_event_to_row(event)])

    loaded_gdf = sta_gdf
    if loaded_gdf is None:
        loaded_gdf, _stats = get_cached_sta_gdf(cfg)

    if loaded_gdf is None:
        result = unavailable_sta_result(event_id, source_missing=True)
        apply_sta_result(event, result)
        session.flush()
        logger.info(
            "Phase 8 STA: event=%s source_missing=True status=%s",
            event_id,
            result.sta_association_status,
        )
        return result

    result = process_event_sta(events_df, event_id, config=cfg, sta_gdf=loaded_gdf)
    apply_sta_result(event, result)
    session.flush()
    logger.info(
        "Phase 8 STA: event=%s status=%s available=%s matches=%s quality=%s",
        event_id,
        result.sta_association_status,
        result.sta_evidence_available,
        result.sta_match_count,
        result.sta_evidence_quality,
    )
    return result
