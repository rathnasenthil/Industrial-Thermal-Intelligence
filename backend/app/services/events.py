"""Event query/filter helpers and response mapping."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.thermal_event import ThermalEvent
from app.schemas.common import pagination_totals, parse_bbox, point_from_lon_lat
from app.schemas.events import (
    EventDetail,
    EventEvidence,
    EventSummary,
    EventTimeline,
    EvidenceFamilyBlock,
    FacilityCandidateSummary,
    PaginatedAlerts,
    PaginatedEvents,
)


def _apply_event_filters(
    stmt: Select,
    *,
    priority: Optional[str] = None,
    industrial_context: Optional[str] = None,
    facility_type: Optional[str] = None,
    persistence_class: Optional[str] = None,
    anomaly_status: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_risk_score: Optional[float] = None,
    max_risk_score: Optional[float] = None,
    bbox: Optional[str] = None,
    facility_id: Optional[str] = None,
    priorities: Optional[list[str]] = None,
) -> Select:
    if priority:
        stmt = stmt.where(ThermalEvent.investigation_priority == priority)
    if priorities:
        stmt = stmt.where(ThermalEvent.investigation_priority.in_(priorities))
    if industrial_context:
        stmt = stmt.where(ThermalEvent.industrial_context == industrial_context)
    if facility_type:
        stmt = stmt.where(ThermalEvent.facility_type == facility_type)
    if persistence_class:
        stmt = stmt.where(ThermalEvent.persistence_label == persistence_class)
    if anomaly_status:
        stmt = stmt.where(ThermalEvent.anomaly_status == anomaly_status)
    if date_from is not None:
        stmt = stmt.where(ThermalEvent.event_start >= date_from)
    if date_to is not None:
        stmt = stmt.where(ThermalEvent.event_start <= date_to)
    if min_risk_score is not None:
        stmt = stmt.where(ThermalEvent.risk_score >= min_risk_score)
    if max_risk_score is not None:
        stmt = stmt.where(ThermalEvent.risk_score <= max_risk_score)
    if facility_id:
        stmt = stmt.where(ThermalEvent.facility_id == facility_id)
    if bbox:
        box = parse_bbox(bbox)
        envelope = func.ST_MakeEnvelope(
            box.min_lon, box.min_lat, box.max_lon, box.max_lat, 4326
        )
        stmt = stmt.where(func.ST_Intersects(ThermalEvent.geometry, envelope))
    return stmt


def event_to_summary(event: ThermalEvent) -> EventSummary:
    return EventSummary(
        event_id=event.event_id,
        event_start=event.event_start,
        event_end=event.event_end,
        observed_duration_hours=event.observed_duration_hours,
        detection_count=event.detection_count,
        peak_frp=event.peak_frp,
        mean_frp=event.mean_frp,
        latitude=event.centroid_latitude,
        longitude=event.centroid_longitude,
        geometry=point_from_lon_lat(event.centroid_longitude, event.centroid_latitude),
        persistence_label=event.persistence_label,
        facility_id=event.facility_id,
        facility_name=event.facility_name,
        facility_type=event.facility_type,
        facility_association_method=event.facility_association_method,
        facility_distance_km=event.facility_distance_km,
        anomaly_status=event.anomaly_status,
        industrial_context=event.industrial_context,
        risk_score=event.risk_score,
        investigation_priority=event.investigation_priority,
        thermal_severity_band=event.thermal_severity_band,
        recommended_action=event.recommended_action,
    )


def list_events(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 50,
    priority: Optional[str] = None,
    industrial_context: Optional[str] = None,
    facility_type: Optional[str] = None,
    persistence_class: Optional[str] = None,
    anomaly_status: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_risk_score: Optional[float] = None,
    max_risk_score: Optional[float] = None,
    bbox: Optional[str] = None,
    facility_id: Optional[str] = None,
    priorities: Optional[list[str]] = None,
) -> PaginatedEvents:
    page = max(page, 1)
    page_size = min(max(page_size, 1), 500)

    base = select(ThermalEvent)
    base = _apply_event_filters(
        base,
        priority=priority,
        industrial_context=industrial_context,
        facility_type=facility_type,
        persistence_class=persistence_class,
        anomaly_status=anomaly_status,
        date_from=date_from,
        date_to=date_to,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
        bbox=bbox,
        facility_id=facility_id,
        priorities=priorities,
    )

    count_stmt = select(func.count()).select_from(base.subquery())
    total = int(db.scalar(count_stmt) or 0)

    stmt = (
        base.order_by(
            ThermalEvent.risk_score.desc().nullslast(),
            ThermalEvent.event_start.desc().nullslast(),
            ThermalEvent.event_id.asc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = list(db.scalars(stmt).all())
    meta = pagination_totals(total, page, page_size)
    return PaginatedEvents(
        items=[event_to_summary(r) for r in rows],
        total=meta.total,
        page=meta.page,
        page_size=meta.page_size,
        total_pages=meta.total_pages,
    )


def get_event(db: Session, event_id: str) -> Optional[EventDetail]:
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        return None

    candidates = list(
        db.scalars(
            select(EventFacilityCandidate)
            .where(EventFacilityCandidate.event_id == event_id)
            .order_by(
                EventFacilityCandidate.candidate_rank.asc().nullslast(),
                EventFacilityCandidate.distance_km.asc().nullslast(),
            )
        ).all()
    )
    summary = event_to_summary(event)
    return EventDetail(
        **summary.model_dump(),
        distinct_detection_days=event.distinct_detection_days,
        span_days=event.span_days,
        duty_cycle=event.duty_cycle,
        mean_gap_hours=event.mean_gap_hours,
        max_gap_hours=event.max_gap_hours,
        median_frp=event.median_frp,
        total_frp=event.total_frp,
        day_detection_count=event.day_detection_count,
        night_detection_count=event.night_detection_count,
        min_latitude=event.min_latitude,
        max_latitude=event.max_latitude,
        min_longitude=event.min_longitude,
        max_longitude=event.max_longitude,
        centroid_wkt=event.centroid_wkt,
        footprint_wkt=event.footprint_wkt,
        persistence_basis=event.persistence_basis,
        facility_attribution_confidence=event.facility_attribution_confidence,
        candidate_facility_count=event.candidate_facility_count,
        facility_candidates=[
            FacilityCandidateSummary.model_validate(c) for c in candidates
        ],
        baseline_observation_count=event.baseline_observation_count,
        baseline_history_status=event.baseline_history_status,
        anomaly_unavailable_reason=event.anomaly_unavailable_reason,
        anomaly_score=event.anomaly_score,
        anomaly_confidence=event.anomaly_confidence,
        anomaly_explanation=event.anomaly_explanation,
        evidence_sufficiency=event.evidence_sufficiency,
        evidence_uncertainty=event.evidence_uncertainty,
        evidence_strength=event.evidence_strength,
        industrial_evidence_score=event.industrial_evidence_score,
        evidence_fusion_score=event.evidence_fusion_score,
        source_intelligence_candidate=event.source_intelligence_candidate,
        candidate_rationale=event.candidate_rationale,
        candidate_is_ground_truth=event.candidate_is_ground_truth,
        interpretation_confidence=event.interpretation_confidence,
        thermal_severity_score=event.thermal_severity_score,
        uncertainty_score=event.uncertainty_score,
        uncertainty_band=event.uncertainty_band,
        dominant_risk_factors=event.dominant_risk_factors,
        dominant_uncertainty_factors=event.dominant_uncertainty_factors,
        priority_reasons=event.priority_reasons,
        priority_warnings=event.priority_warnings,
        risk_limiting_evidence_codes=event.risk_limiting_evidence_codes,
        risk_scoring_version=event.risk_scoring_version,
    )


def _family(
    *,
    available: Optional[bool],
    score: Optional[float],
    summary: Optional[str],
    details: dict,
) -> EvidenceFamilyBlock:
    is_available = bool(available)
    return EvidenceFamilyBlock(
        available=is_available,
        status="available" if is_available else "unavailable",
        score=score if is_available else None,
        summary=summary,
        details=details,
    )


def get_event_evidence(db: Session, event_id: str) -> Optional[EventEvidence]:
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        return None

    # Historical evidence is derived from facility baseline history when present.
    historical_available = (
        event.baseline_history_status is not None
        and event.baseline_history_status not in {"NO_PRIOR_OBSERVATIONS"}
    ) or (event.historical_evidence_score is not None)

    anomaly_available = event.anomaly_status not in {
        None,
        "INSUFFICIENT_HISTORY",
        "UNAVAILABLE",
    }

    sta_available = bool(event.sta_domain_available)
    env_available = bool(event.environmental_domain_available)

    return EventEvidence(
        event_id=event.event_id,
        temporal=_family(
            available=event.temporal_evidence_available,
            score=event.temporal_evidence_score,
            summary=event.temporal_evidence_summary,
            details={
                "temporal_persistence_signal": event.temporal_persistence_signal,
                "temporal_anomaly_signal": event.temporal_anomaly_signal,
                "persistence_label": event.persistence_label,
            },
        ),
        infrastructure=_family(
            available=event.infrastructure_evidence_available,
            score=event.infrastructure_evidence_score,
            summary=event.infrastructure_evidence_summary,
            details={
                "infrastructure_association_signal": event.infrastructure_association_signal,
                "infrastructure_facility_type_signal": event.infrastructure_facility_type_signal,
                "infrastructure_confidence_signal": event.infrastructure_confidence_signal,
                "infrastructure_history_signal": event.infrastructure_history_signal,
                "facility_association_method": event.facility_association_method,
            },
        ),
        historical=_family(
            available=historical_available,
            score=event.historical_evidence_score,
            summary=event.baseline_history_status,
            details={
                "baseline_observation_count": event.baseline_observation_count,
                "baseline_history_status": event.baseline_history_status,
            },
        ),
        anomaly=_family(
            available=anomaly_available,
            score=event.anomaly_evidence_score if anomaly_available else None,
            summary=event.anomaly_explanation,
            details={
                "anomaly_status": event.anomaly_status,
                "anomaly_score": event.anomaly_score,
                "anomaly_confidence": event.anomaly_confidence,
                "anomaly_unavailable_reason": event.anomaly_unavailable_reason,
            },
        ),
        sta=_family(
            available=sta_available,
            score=event.sta_evidence_score if sta_available else None,
            summary=(
                event.sta_evidence_summary
                if sta_available
                else "STA domain unavailable in Stage VI source data; not treated as negative evidence."
            ),
            details={
                "sta_domain_available": event.sta_domain_available,
                "sta_association_signal": event.sta_association_signal,
                "sta_layer_signal": event.sta_layer_signal,
                "sta_quality_signal": event.sta_quality_signal,
            },
        ),
        environmental=_family(
            available=env_available,
            score=event.environmental_evidence_score if env_available else None,
            summary=(
                event.environmental_evidence_summary
                if env_available
                else "Environmental domain unavailable in Stage VI source data; "
                "not treated as negative evidence."
            ),
            details={
                "environmental_domain_available": event.environmental_domain_available,
                "landcover_available": event.landcover_available,
                "vegetation_context_available": event.vegetation_context_available,
                "builtup_context_available": event.builtup_context_available,
                "water_context_available": event.water_context_available,
                "agriculture_context_available": event.agriculture_context_available,
                "satellite_context_available": event.satellite_context_available,
                "environmental_landcover_signal": event.environmental_landcover_signal,
                "environmental_vegetation_signal": event.environmental_vegetation_signal,
                "environmental_agriculture_signal": event.environmental_agriculture_signal,
                "environmental_builtup_signal": event.environmental_builtup_signal,
                "environmental_water_signal": event.environmental_water_signal,
            },
        ),
        fusion={
            "industrial_evidence_score": event.industrial_evidence_score,
            "evidence_fusion_score": event.evidence_fusion_score,
            "evidence_coverage": event.evidence_coverage,
            "evidence_strength": event.evidence_strength,
            "evidence_sufficiency": event.evidence_sufficiency,
            "evidence_uncertainty": event.evidence_uncertainty,
            "source_intelligence_candidate": event.source_intelligence_candidate,
            "candidate_rationale": event.candidate_rationale,
            "candidate_is_ground_truth": event.candidate_is_ground_truth,
            "interpretation_confidence": event.interpretation_confidence,
            "evidence_sources_present": event.evidence_sources_present,
            "evidence_sources_missing": event.evidence_sources_missing,
            "evidence_availability_summary": event.evidence_availability_summary,
            "evidence_conflict_flag": event.evidence_conflict_flag,
            "evidence_conflict_codes": event.evidence_conflict_codes,
            "supporting_evidence_codes": event.supporting_evidence_codes,
            "ambiguous_evidence_codes": event.ambiguous_evidence_codes,
            "limiting_evidence_codes": event.limiting_evidence_codes,
            "note": (
                "Candidate labels are interpretations, not ground truth. "
                "Stage V produced no validated performance claim."
            ),
        },
    )


def get_event_timeline(db: Session, event_id: str) -> Optional[EventTimeline]:
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        return None
    return EventTimeline(
        event_id=event.event_id,
        event_start=event.event_start,
        event_end=event.event_end,
        observed_duration_hours=event.observed_duration_hours,
        distinct_detection_days=event.distinct_detection_days,
        span_days=event.span_days,
        duty_cycle=event.duty_cycle,
        mean_gap_hours=event.mean_gap_hours,
        max_gap_hours=event.max_gap_hours,
        detection_count=event.detection_count,
        day_detection_count=event.day_detection_count,
        night_detection_count=event.night_detection_count,
        detection_level_timeline_available=False,
    )


def list_alerts(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 50,
    industrial_context: Optional[str] = None,
    facility_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_risk_score: Optional[float] = None,
    max_risk_score: Optional[float] = None,
    bbox: Optional[str] = None,
) -> PaginatedAlerts:
    """
    Investigation-priority view of HIGH/CRITICAL events.

    This is not an emergency dispatch or push-notification system.
    """
    result = list_events(
        db,
        page=page,
        page_size=page_size,
        industrial_context=industrial_context,
        facility_type=facility_type,
        date_from=date_from,
        date_to=date_to,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
        bbox=bbox,
        priorities=["HIGH", "CRITICAL"],
    )
    return PaginatedAlerts(
        items=result.items,
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )
