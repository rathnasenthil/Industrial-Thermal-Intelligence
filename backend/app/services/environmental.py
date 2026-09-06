"""
Backend Phase 9: incremental Stage I.6 Environmental / Satellite Context.

Loads one ThermalEvent geometry context, calls AIML ``realtime.environmental``,
and writes I.6 columns onto **that event only**.

Environmental context is evidence/context only — not ground truth /
industrial-fire classification / anomaly labels / risk scoring.

Does **not** call ``run_environmental_context()`` over all historical events.
Does **not** fabricate environmental values when local datasets are absent.
Does **not** modify I.3 fingerprint tables, I.4 anomaly fields, or I.5 STA fields.
Does **not** write Stage I.7 ``environmental_*_signal`` fusion columns.
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

from realtime.environmental import (  # noqa: E402
    EnvironmentalContextResult,
    any_environmental_source_present,
    process_event_environmental,
    unavailable_environmental_result,
)
from src.environmental_context.config import (  # noqa: E402
    DEFAULT_CONFIG,
    EnvironmentalContextConfig,
)
from src.environmental_context.context_schema import ALL_CONTEXT_COLUMNS  # noqa: E402


@dataclass
class EnvironmentalContextStats:
    events_updated: int = 0
    event_ids: list[str] = field(default_factory=list)
    by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _abs_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return _AIML_ROOT / p


def aiml_environmental_config(
    config: Optional[EnvironmentalContextConfig] = None,
) -> EnvironmentalContextConfig:
    """Resolve I.6 dataset paths relative to the aiml/ package root."""
    base = config or DEFAULT_CONFIG
    return EnvironmentalContextConfig(
        context_buffer_km=base.context_buffer_km,
        broad_context_buffer_km=base.broad_context_buffer_km,
        events_path=_abs_path(base.events_path) or base.events_path,
        events_fallback_path=_abs_path(base.events_fallback_path)
        or base.events_fallback_path,
        landcover_raster_path=_abs_path(base.landcover_raster_path),
        landcover_vector_path=_abs_path(base.landcover_vector_path),
        vegetation_path=_abs_path(base.vegetation_path),
        builtup_path=_abs_path(base.builtup_path),
        water_path=_abs_path(base.water_path),
        agriculture_path=_abs_path(base.agriculture_path),
        satellite_raster_path=_abs_path(base.satellite_raster_path),
        landcover_class_map=dict(base.landcover_class_map),
        landcover_year=base.landcover_year,
        landcover_source_name=base.landcover_source_name,
    )


def _event_to_row(event: ThermalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_start": event.event_start,
        "event_end": event.event_end,
        "footprint_wkt": event.footprint_wkt,
        "centroid_wkt": event.centroid_wkt,
        "centroid_latitude": event.centroid_latitude,
        "centroid_longitude": event.centroid_longitude,
        # I.4 / I.5 present so batch immutability checks can compare if present;
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
        "sta_association_status": event.sta_association_status,
        "sta_evidence_available": event.sta_evidence_available,
        "sta_evidence_quality": event.sta_evidence_quality,
        "sta_match_count": event.sta_match_count,
        "primary_sta_id": event.primary_sta_id,
    }


def apply_environmental_result(
    event: ThermalEvent, result: EnvironmentalContextResult
) -> None:
    """Write I.6 columns onto one ThermalEvent (does not touch I.3/I.4/I.5/I.7)."""
    event.landcover_available = result.landcover_available
    event.landcover_source = result.landcover_source
    event.landcover_year = result.landcover_year
    event.dominant_landcover_class = result.dominant_landcover_class
    event.dominant_landcover_fraction = result.dominant_landcover_fraction
    event.landcover_class_count = (
        int(result.landcover_class_count)
        if result.landcover_class_count is not None
        else None
    )
    event.vegetation_context_available = result.vegetation_context_available
    event.vegetation_present = result.vegetation_present
    event.vegetation_coverage_fraction = result.vegetation_coverage_fraction
    event.distance_to_vegetation_km = result.distance_to_vegetation_km
    event.builtup_context_available = result.builtup_context_available
    event.builtup_present = result.builtup_present
    event.builtup_coverage_fraction = result.builtup_coverage_fraction
    event.distance_to_builtup_km = result.distance_to_builtup_km
    event.water_context_available = result.water_context_available
    event.water_present = result.water_present
    event.water_coverage_fraction = result.water_coverage_fraction
    event.distance_to_water_km = result.distance_to_water_km
    event.agriculture_context_available = result.agriculture_context_available
    event.agriculture_present = result.agriculture_present
    event.agriculture_coverage_fraction = result.agriculture_coverage_fraction
    event.distance_to_agriculture_km = result.distance_to_agriculture_km
    event.satellite_context_available = result.satellite_context_available
    event.satellite_source = result.satellite_source
    event.satellite_value = result.satellite_value
    event.satellite_value_name = result.satellite_value_name


def refresh_event_environmental(
    session: Session,
    event_id: str,
    *,
    config: Optional[EnvironmentalContextConfig] = None,
) -> EnvironmentalContextResult:
    """
    Phase 9 entry: recompute I.6 for one event after Phase 8.

    When local environmental datasets are absent, writes unavailable defaults
    (same as batch empty_like_unavailable) and sets ``source_missing``.
    """
    cfg = aiml_environmental_config(config)
    event = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        raise RuntimeError(f"Event missing for environmental refresh: {event_id}")

    ensure_event_geometry_wkt(session, event)
    session.refresh(event)

    if event.centroid_latitude is None or event.centroid_longitude is None:
        result = unavailable_environmental_result(event_id, source_missing=False)
        apply_environmental_result(event, result)
        session.flush()
        return result

    events_df = pd.DataFrame([_event_to_row(event)])

    if not any_environmental_source_present(cfg):
        result = unavailable_environmental_result(event_id, source_missing=True)
        apply_environmental_result(event, result)
        session.flush()
        logger.info(
            "Phase 9 environmental: event=%s source_missing=True "
            "landcover=%s water=%s",
            event_id,
            result.landcover_available,
            result.water_context_available,
        )
        return result

    result = process_event_environmental(events_df, event_id, config=cfg)
    apply_environmental_result(event, result)
    session.flush()
    logger.info(
        "Phase 9 environmental: event=%s landcover=%s veg=%s builtup=%s "
        "water=%s agri=%s sat=%s source_missing=%s",
        event_id,
        result.landcover_available,
        result.vegetation_context_available,
        result.builtup_context_available,
        result.water_context_available,
        result.agriculture_context_available,
        result.satellite_context_available,
        result.source_missing,
    )
    # Sanity: applied columns ⊆ batch schema
    _ = ALL_CONTEXT_COLUMNS
    return result
