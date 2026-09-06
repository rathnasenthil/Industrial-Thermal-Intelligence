"""
Bulk ingestion of frozen AIML Stage VI / I.1 / I.2 outputs into PostGIS.

Does not re-run AIML logic. Preserves null/unavailable evidence semantics.
Idempotent via truncate-and-reload (development) or controlled replace.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from geoalchemy2.elements import WKTElement
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility import Facility
from app.models.thermal_event import ThermalEvent
from app.services.parsing import (
    parse_json_object,
    parse_optional_bool,
    parse_optional_float,
    parse_optional_int,
    parse_optional_str,
    parse_timestamp,
    point_wkt,
    valid_lon_lat,
)

logger = logging.getLogger(__name__)

DEFAULT_EVENT_CSV = Path(__file__).resolve().parents[3] / (
    "aiml/data/processed/thermal_events_with_risk_prioritization.csv"
)
DEFAULT_FACILITY_CSV = Path(__file__).resolve().parents[3] / (
    "aiml/data/processed/osm_facilities.csv"
)
DEFAULT_CANDIDATE_CSV = Path(__file__).resolve().parents[3] / (
    "aiml/data/processed/thermal_event_facility_candidates.csv"
)

REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "event_start",
    "event_end",
    "centroid_latitude",
    "centroid_longitude",
    "detection_count",
    "peak_frp",
    "mean_frp",
    "persistence_label",
    "risk_score",
    "investigation_priority",
    "industrial_context",
    "recommended_action",
}

REQUIRED_FACILITY_COLUMNS = {
    "facility_id",
    "facility_name",
    "facility_type",
    "latitude",
    "longitude",
    "osm_id",
    "osm_type",
    "source",
    "source_version",
}

REQUIRED_CANDIDATE_COLUMNS = {
    "event_id",
    "facility_id",
    "spatial_relation",
    "distance_km",
    "candidate_rank",
}

BATCH_SIZE = 2_000


@dataclass
class IngestionReport:
    facilities_source_rows: int = 0
    facilities_inserted: int = 0
    facilities_rejected: int = 0
    events_source_rows: int = 0
    events_inserted: int = 0
    events_rejected: int = 0
    candidates_source_rows: int = 0
    candidates_inserted: int = 0
    candidates_rejected: int = 0
    duplicate_event_ids: int = 0
    duplicate_facility_ids: int = 0
    rejection_samples: list[str] = field(default_factory=list)
    mode: str = "replace"
    errors: list[str] = field(default_factory=list)

    def add_rejection(self, message: str, *, limit: int = 50) -> None:
        if len(self.rejection_samples) < limit:
            self.rejection_samples.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "facilities_source_rows": self.facilities_source_rows,
            "facilities_inserted": self.facilities_inserted,
            "facilities_rejected": self.facilities_rejected,
            "events_source_rows": self.events_source_rows,
            "events_inserted": self.events_inserted,
            "events_rejected": self.events_rejected,
            "candidates_source_rows": self.candidates_source_rows,
            "candidates_inserted": self.candidates_inserted,
            "candidates_rejected": self.candidates_rejected,
            "duplicate_event_ids": self.duplicate_event_ids,
            "duplicate_facility_ids": self.duplicate_facility_ids,
            "rejection_samples": self.rejection_samples,
            "errors": self.errors,
        }


def _validate_columns(fieldnames: Optional[list[str]], required: set[str], label: str) -> None:
    if not fieldnames:
        raise ValueError(f"{label} CSV has no header row")
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{label} CSV missing required columns: {missing}")


def _geometry_point(lon: Optional[float], lat: Optional[float]) -> Optional[WKTElement]:
    if not valid_lon_lat(lon, lat):
        return None
    assert lon is not None and lat is not None
    return WKTElement(point_wkt(lon, lat), srid=4326)


def _geometry_from_wkt(wkt: Optional[str]) -> Optional[WKTElement]:
    if not wkt:
        return None
    text = wkt.strip()
    if not text:
        return None
    return WKTElement(text, srid=4326)


def _facility_row(raw: dict[str, str]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    facility_id = parse_optional_str(raw.get("facility_id"))
    if not facility_id:
        return None, "missing facility_id"
    lat = parse_optional_float(raw.get("latitude"))
    lon = parse_optional_float(raw.get("longitude"))
    if not valid_lon_lat(lon, lat):
        return None, f"invalid coordinates for {facility_id}"
    return {
        "facility_id": facility_id,
        "facility_name": parse_optional_str(raw.get("facility_name")),
        "facility_type": parse_optional_str(raw.get("facility_type")),
        "industrial_subtype": parse_optional_str(raw.get("industrial_subtype")),
        "operator": parse_optional_str(raw.get("operator")),
        "landuse": parse_optional_str(raw.get("landuse")),
        "power_type": parse_optional_str(raw.get("power_type")),
        "man_made_type": parse_optional_str(raw.get("man_made_type")),
        "confidence": parse_optional_str(raw.get("confidence")),
        "geometry_type": parse_optional_str(raw.get("geometry_type")),
        "latitude": lat,
        "longitude": lon,
        "geometry": _geometry_point(lon, lat),
        "geometry_wkt": parse_optional_str(raw.get("geometry_wkt")),
        "osm_id": parse_optional_str(raw.get("osm_id")),
        "osm_type": parse_optional_str(raw.get("osm_type")),
        "osm_tags": parse_json_object(raw.get("osm_tags")),
        "source": parse_optional_str(raw.get("source")),
        "source_version": parse_optional_str(raw.get("source_version")),
    }, None


def _event_row(raw: dict[str, str]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    event_id = parse_optional_str(raw.get("event_id"))
    if not event_id:
        return None, "missing event_id"

    lat = parse_optional_float(raw.get("centroid_latitude"))
    lon = parse_optional_float(raw.get("centroid_longitude"))
    if not valid_lon_lat(lon, lat):
        return None, f"invalid centroid coordinates for {event_id}"

    risk_score = parse_optional_float(raw.get("risk_score"))
    if risk_score is not None and not (0.0 <= risk_score <= 100.0):
        return None, f"risk_score out of range for {event_id}: {risk_score}"

    footprint_wkt = parse_optional_str(raw.get("footprint_wkt"))
    return {
        "event_id": event_id,
        "event_start": parse_timestamp(raw.get("event_start")),
        "event_end": parse_timestamp(raw.get("event_end")),
        "observed_duration_hours": parse_optional_float(raw.get("observed_duration_hours")),
        "distinct_detection_days": parse_optional_int(raw.get("distinct_detection_days")),
        "span_days": parse_optional_float(raw.get("span_days")),
        "duty_cycle": parse_optional_float(raw.get("duty_cycle")),
        "mean_gap_hours": parse_optional_float(raw.get("mean_gap_hours")),
        "max_gap_hours": parse_optional_float(raw.get("max_gap_hours")),
        "detection_frequency_per_day": parse_optional_float(
            raw.get("detection_frequency_per_day")
        ),
        "detection_count": parse_optional_int(raw.get("detection_count")),
        "peak_frp": parse_optional_float(raw.get("peak_frp")),
        "mean_frp": parse_optional_float(raw.get("mean_frp")),
        "median_frp": parse_optional_float(raw.get("median_frp")),
        "total_frp": parse_optional_float(raw.get("total_frp")),
        "day_detection_count": parse_optional_int(raw.get("day_detection_count")),
        "night_detection_count": parse_optional_int(raw.get("night_detection_count")),
        "centroid_latitude": lat,
        "centroid_longitude": lon,
        "min_latitude": parse_optional_float(raw.get("min_latitude")),
        "max_latitude": parse_optional_float(raw.get("max_latitude")),
        "min_longitude": parse_optional_float(raw.get("min_longitude")),
        "max_longitude": parse_optional_float(raw.get("max_longitude")),
        "centroid_wkt": parse_optional_str(raw.get("centroid_wkt")),
        "footprint_wkt": footprint_wkt,
        "geometry": _geometry_point(lon, lat),
        "footprint_geometry": _geometry_from_wkt(footprint_wkt),
        "persistence_label": parse_optional_str(raw.get("persistence_label")),
        "persistence_basis": parse_optional_str(raw.get("persistence_basis")),
        "facility_id": parse_optional_str(raw.get("facility_id")),
        "facility_name": parse_optional_str(raw.get("facility_name")),
        "facility_type": parse_optional_str(raw.get("facility_type")),
        "facility_association_method": parse_optional_str(
            raw.get("facility_association_method")
        ),
        "facility_attribution_confidence": parse_optional_str(
            raw.get("facility_attribution_confidence")
        ),
        "facility_distance_km": parse_optional_float(raw.get("facility_distance_km")),
        "candidate_facility_count": parse_optional_int(raw.get("candidate_facility_count")),
        "baseline_observation_count": parse_optional_int(raw.get("baseline_observation_count")),
        "baseline_history_status": parse_optional_str(raw.get("baseline_history_status")),
        "anomaly_unavailable_reason": parse_optional_str(raw.get("anomaly_unavailable_reason")),
        "anomaly_score": parse_optional_float(raw.get("anomaly_score")),
        "anomaly_status": parse_optional_str(raw.get("anomaly_status")),
        "anomaly_confidence": parse_optional_str(raw.get("anomaly_confidence")),
        "peak_frp_deviation": parse_optional_float(raw.get("peak_frp_deviation")),
        "event_size_deviation": parse_optional_float(raw.get("event_size_deviation")),
        "duration_deviation": parse_optional_float(raw.get("duration_deviation")),
        "distance_deviation": parse_optional_float(raw.get("distance_deviation")),
        "persistence_deviation": parse_optional_float(raw.get("persistence_deviation")),
        "monthly_deviation": parse_optional_float(raw.get("monthly_deviation")),
        "features_available": parse_optional_int(raw.get("features_available")),
        "features_evaluated": parse_optional_int(raw.get("features_evaluated")),
        "anomaly_explanation": parse_optional_str(raw.get("anomaly_explanation")),
        "landcover_available": parse_optional_bool(raw.get("landcover_available")),
        "vegetation_context_available": parse_optional_bool(
            raw.get("vegetation_context_available")
        ),
        "builtup_context_available": parse_optional_bool(raw.get("builtup_context_available")),
        "water_context_available": parse_optional_bool(raw.get("water_context_available")),
        "agriculture_context_available": parse_optional_bool(
            raw.get("agriculture_context_available")
        ),
        "satellite_context_available": parse_optional_bool(
            raw.get("satellite_context_available")
        ),
        "temporal_evidence_available": parse_optional_bool(
            raw.get("temporal_evidence_available")
        ),
        "infrastructure_evidence_available": parse_optional_bool(
            raw.get("infrastructure_evidence_available")
        ),
        "sta_domain_available": parse_optional_bool(raw.get("sta_domain_available")),
        "environmental_domain_available": parse_optional_bool(
            raw.get("environmental_domain_available")
        ),
        "temporal_persistence_signal": parse_optional_str(
            raw.get("temporal_persistence_signal")
        ),
        "temporal_anomaly_signal": parse_optional_str(raw.get("temporal_anomaly_signal")),
        "temporal_evidence_summary": parse_optional_str(raw.get("temporal_evidence_summary")),
        "infrastructure_association_signal": parse_optional_str(
            raw.get("infrastructure_association_signal")
        ),
        "infrastructure_facility_type_signal": parse_optional_str(
            raw.get("infrastructure_facility_type_signal")
        ),
        "infrastructure_confidence_signal": parse_optional_str(
            raw.get("infrastructure_confidence_signal")
        ),
        "infrastructure_history_signal": parse_optional_str(
            raw.get("infrastructure_history_signal")
        ),
        "infrastructure_evidence_summary": parse_optional_str(
            raw.get("infrastructure_evidence_summary")
        ),
        "sta_association_signal": parse_optional_str(raw.get("sta_association_signal")),
        "sta_layer_signal": parse_optional_str(raw.get("sta_layer_signal")),
        "sta_quality_signal": parse_optional_str(raw.get("sta_quality_signal")),
        "sta_evidence_summary": parse_optional_str(raw.get("sta_evidence_summary")),
        "environmental_landcover_signal": parse_optional_str(
            raw.get("environmental_landcover_signal")
        ),
        "environmental_vegetation_signal": parse_optional_str(
            raw.get("environmental_vegetation_signal")
        ),
        "environmental_agriculture_signal": parse_optional_str(
            raw.get("environmental_agriculture_signal")
        ),
        "environmental_builtup_signal": parse_optional_str(
            raw.get("environmental_builtup_signal")
        ),
        "environmental_water_signal": parse_optional_str(
            raw.get("environmental_water_signal")
        ),
        "environmental_evidence_summary": parse_optional_str(
            raw.get("environmental_evidence_summary")
        ),
        "evidence_sources_present_count": parse_optional_int(
            raw.get("evidence_sources_present_count")
        ),
        "evidence_sources_present": parse_optional_str(raw.get("evidence_sources_present")),
        "evidence_sources_missing": parse_optional_str(raw.get("evidence_sources_missing")),
        "evidence_availability_summary": parse_optional_str(
            raw.get("evidence_availability_summary")
        ),
        "evidence_conflict_flag": parse_optional_bool(raw.get("evidence_conflict_flag")),
        "evidence_conflict_codes": parse_optional_str(raw.get("evidence_conflict_codes")),
        "evidence_conflict_summary": parse_optional_str(raw.get("evidence_conflict_summary")),
        "infrastructure_evidence_score": parse_optional_float(
            raw.get("infrastructure_evidence_score")
        ),
        "temporal_evidence_score": parse_optional_float(raw.get("temporal_evidence_score")),
        "historical_evidence_score": parse_optional_float(raw.get("historical_evidence_score")),
        "anomaly_evidence_score": parse_optional_float(raw.get("anomaly_evidence_score")),
        "sta_evidence_score": parse_optional_float(raw.get("sta_evidence_score")),
        "environmental_evidence_score": parse_optional_float(
            raw.get("environmental_evidence_score")
        ),
        "industrial_evidence_score": parse_optional_float(raw.get("industrial_evidence_score")),
        "environmental_support_score": parse_optional_float(
            raw.get("environmental_support_score")
        ),
        "evidence_fusion_score": parse_optional_float(raw.get("evidence_fusion_score")),
        "evidence_coverage": parse_optional_str(raw.get("evidence_coverage")),
        "evidence_strength": parse_optional_str(raw.get("evidence_strength")),
        "evidence_profile_codes": parse_optional_str(raw.get("evidence_profile_codes")),
        "supporting_evidence_codes": parse_optional_str(raw.get("supporting_evidence_codes")),
        "ambiguous_evidence_codes": parse_optional_str(raw.get("ambiguous_evidence_codes")),
        "limiting_evidence_codes": parse_optional_str(raw.get("limiting_evidence_codes")),
        "source_intelligence_candidate": parse_optional_str(
            raw.get("source_intelligence_candidate")
        ),
        "candidate_rationale": parse_optional_str(raw.get("candidate_rationale")),
        "candidate_is_ground_truth": parse_optional_bool(raw.get("candidate_is_ground_truth")),
        "evidence_sufficiency": parse_optional_str(raw.get("evidence_sufficiency")),
        "evidence_uncertainty": parse_optional_str(raw.get("evidence_uncertainty")),
        "interpretation_confidence": parse_optional_str(raw.get("interpretation_confidence")),
        "risk_score": risk_score,
        "investigation_priority": parse_optional_str(raw.get("investigation_priority")),
        "recommended_action": parse_optional_str(raw.get("recommended_action")),
        "industrial_context": parse_optional_str(raw.get("industrial_context")),
        "thermal_severity_score": parse_optional_float(raw.get("thermal_severity_score")),
        "thermal_severity_band": parse_optional_str(raw.get("thermal_severity_band")),
        "persistence_priority_score": parse_optional_float(
            raw.get("persistence_priority_score")
        ),
        "persistence_priority_reason": parse_optional_str(
            raw.get("persistence_priority_reason")
        ),
        "anomaly_priority_score": parse_optional_float(raw.get("anomaly_priority_score")),
        "anomaly_priority_reason": parse_optional_str(raw.get("anomaly_priority_reason")),
        "facility_context_score": parse_optional_float(raw.get("facility_context_score")),
        "facility_context_reason": parse_optional_str(raw.get("facility_context_reason")),
        "industrial_evidence_component": parse_optional_float(
            raw.get("industrial_evidence_component")
        ),
        "uncertainty_score": parse_optional_float(raw.get("uncertainty_score")),
        "uncertainty_band": parse_optional_str(raw.get("uncertainty_band")),
        "dominant_risk_factors": parse_optional_str(raw.get("dominant_risk_factors")),
        "dominant_uncertainty_factors": parse_optional_str(
            raw.get("dominant_uncertainty_factors")
        ),
        "priority_reasons": parse_optional_str(raw.get("priority_reasons")),
        "priority_warnings": parse_optional_str(raw.get("priority_warnings")),
        "risk_limiting_evidence_codes": parse_optional_str(
            raw.get("risk_limiting_evidence_codes")
        ),
        "risk_scoring_version": parse_optional_str(raw.get("risk_scoring_version")),
    }, None


def _candidate_row(raw: dict[str, str]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    event_id = parse_optional_str(raw.get("event_id"))
    facility_id = parse_optional_str(raw.get("facility_id"))
    if not event_id or not facility_id:
        return None, "missing event_id or facility_id"
    return {
        "event_id": event_id,
        "facility_id": facility_id,
        "facility_name": parse_optional_str(raw.get("facility_name")),
        "facility_type": parse_optional_str(raw.get("facility_type")),
        "spatial_relation": parse_optional_str(raw.get("spatial_relation")),
        "distance_km": parse_optional_float(raw.get("distance_km")),
        "candidate_rank": parse_optional_int(raw.get("candidate_rank")),
        "candidate_score": parse_optional_float(raw.get("candidate_score")),
    }, None


def _flush_batch(session: Session, table, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.execute(pg_insert(table), rows)
    session.flush()


def _iter_csv(path: Path) -> tuple[list[str], Iterable[dict[str, str]]]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"CSV has no header: {path}")
    fieldnames = list(reader.fieldnames)

    def _rows() -> Iterable[dict[str, str]]:
        try:
            yield from reader
        finally:
            handle.close()

    return fieldnames, _rows()


def ingest_facilities(
    session: Session,
    csv_path: Path,
    report: IngestionReport,
) -> set[str]:
    fieldnames, rows = _iter_csv(csv_path)
    _validate_columns(fieldnames, REQUIRED_FACILITY_COLUMNS, "facilities")

    seen: set[str] = set()
    batch: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()

    for raw in rows:
        report.facilities_source_rows += 1
        row, err = _facility_row(raw)
        if err or row is None:
            report.facilities_rejected += 1
            report.add_rejection(f"facility: {err}")
            continue
        fid = row["facility_id"]
        if fid in seen:
            report.duplicate_facility_ids += 1
            report.facilities_rejected += 1
            report.add_rejection(f"duplicate facility_id: {fid}")
            continue
        seen.add(fid)
        accepted_ids.add(fid)
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            _flush_batch(session, Facility.__table__, batch)
            report.facilities_inserted += len(batch)
            batch.clear()

    if batch:
        _flush_batch(session, Facility.__table__, batch)
        report.facilities_inserted += len(batch)
    return accepted_ids


def ingest_events(
    session: Session,
    csv_path: Path,
    report: IngestionReport,
) -> set[str]:
    fieldnames, rows = _iter_csv(csv_path)
    _validate_columns(fieldnames, REQUIRED_EVENT_COLUMNS, "thermal_events")

    seen: set[str] = set()
    batch: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()

    for raw in rows:
        report.events_source_rows += 1
        row, err = _event_row(raw)
        if err or row is None:
            report.events_rejected += 1
            report.add_rejection(f"event: {err}")
            continue
        eid = row["event_id"]
        if eid in seen:
            report.duplicate_event_ids += 1
            report.events_rejected += 1
            report.add_rejection(f"duplicate event_id: {eid}")
            continue
        seen.add(eid)
        accepted_ids.add(eid)
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            _flush_batch(session, ThermalEvent.__table__, batch)
            report.events_inserted += len(batch)
            batch.clear()

    if batch:
        _flush_batch(session, ThermalEvent.__table__, batch)
        report.events_inserted += len(batch)
    return accepted_ids


def ingest_candidates(
    session: Session,
    csv_path: Path,
    report: IngestionReport,
    *,
    valid_event_ids: set[str],
    valid_facility_ids: set[str],
) -> None:
    fieldnames, rows = _iter_csv(csv_path)
    _validate_columns(fieldnames, REQUIRED_CANDIDATE_COLUMNS, "event_facility_candidates")

    seen: set[tuple[str, str]] = set()
    batch: list[dict[str, Any]] = []

    for raw in rows:
        report.candidates_source_rows += 1
        row, err = _candidate_row(raw)
        if err or row is None:
            report.candidates_rejected += 1
            report.add_rejection(f"candidate: {err}")
            continue
        key = (row["event_id"], row["facility_id"])
        if key in seen:
            report.candidates_rejected += 1
            report.add_rejection(f"duplicate candidate pair: {key}")
            continue
        if row["event_id"] not in valid_event_ids:
            report.candidates_rejected += 1
            report.add_rejection(f"candidate event_id not loaded: {row['event_id']}")
            continue
        if row["facility_id"] not in valid_facility_ids:
            report.candidates_rejected += 1
            report.add_rejection(f"candidate facility_id not loaded: {row['facility_id']}")
            continue
        seen.add(key)
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            _flush_batch(session, EventFacilityCandidate.__table__, batch)
            report.candidates_inserted += len(batch)
            batch.clear()

    if batch:
        _flush_batch(session, EventFacilityCandidate.__table__, batch)
        report.candidates_inserted += len(batch)


def run_ingestion(
    session: Session,
    *,
    events_csv: Path = DEFAULT_EVENT_CSV,
    facilities_csv: Path = DEFAULT_FACILITY_CSV,
    candidates_csv: Path = DEFAULT_CANDIDATE_CSV,
    mode: str = "replace",
    load_candidates: bool = True,
) -> IngestionReport:
    """
    Load frozen AIML CSVs into PostgreSQL/PostGIS.

    mode=replace: truncate dependent tables then reload (idempotent full refresh).
    """
    report = IngestionReport(mode=mode)

    for path, label in (
        (facilities_csv, "facilities"),
        (events_csv, "events"),
        (candidates_csv, "candidates"),
    ):
        if not path.exists():
            msg = f"{label} CSV not found: {path}"
            report.errors.append(msg)
            raise FileNotFoundError(msg)

    if mode != "replace":
        raise ValueError(f"Unsupported ingestion mode: {mode}")

    # Order matters: candidates FK → events + facilities.
    session.execute(delete(EventFacilityCandidate))
    session.execute(delete(ThermalEvent))
    session.execute(delete(Facility))
    session.flush()

    logger.info("Ingesting facilities from %s", facilities_csv)
    facility_ids = ingest_facilities(session, facilities_csv, report)
    session.commit()

    logger.info("Ingesting thermal events from %s", events_csv)
    event_ids = ingest_events(session, events_csv, report)
    session.commit()

    if load_candidates:
        logger.info("Ingesting facility candidates from %s", candidates_csv)
        ingest_candidates(
            session,
            candidates_csv,
            report,
            valid_event_ids=event_ids,
            valid_facility_ids=facility_ids,
        )
        session.commit()

    # Integrity checks against DB counts (not hardcoded production expectations).
    db_events = session.scalar(select(func.count()).select_from(ThermalEvent)) or 0
    db_facilities = session.scalar(select(func.count()).select_from(Facility)) or 0
    if db_events != report.events_inserted:
        report.errors.append(
            f"DB event count {db_events} != inserted {report.events_inserted}"
        )
    if db_facilities != report.facilities_inserted:
        report.errors.append(
            f"DB facility count {db_facilities} != inserted {report.facilities_inserted}"
        )
    if report.events_inserted != report.events_source_rows - report.events_rejected:
        report.errors.append(
            "Event insert accounting mismatch "
            f"(source={report.events_source_rows}, rejected={report.events_rejected}, "
            f"inserted={report.events_inserted})"
        )

    # Confirm PostGIS SRID on a sample geometry when available.
    sample_srid = session.execute(
        text(
            "SELECT ST_SRID(geometry) FROM thermal_events "
            "WHERE geometry IS NOT NULL LIMIT 1"
        )
    ).scalar()
    if sample_srid is not None and int(sample_srid) != 4326:
        report.errors.append(f"Unexpected event geometry SRID: {sample_srid}")

    return report
