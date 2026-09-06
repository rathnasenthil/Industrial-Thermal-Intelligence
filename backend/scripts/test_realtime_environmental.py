"""
Manual Phase 1-9 verification: NRT → … → I.5 → I.6 Environmental Context.

NOT a scheduler. Does not run fusion / risk / WebSockets.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_environmental.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
AIML_ROOT = BACKEND_ROOT.parent / "aiml"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(AIML_ROOT))

from sqlalchemy import func, select, text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.database import SessionLocal  # noqa: E402
from app.models.event_detection import EventDetection  # noqa: E402
from app.models.event_facility_candidate import EventFacilityCandidate  # noqa: E402
from app.models.facility import Facility  # noqa: E402
from app.models.firms_observation import FirmsObservation  # noqa: E402
from app.models.thermal_event import ThermalEvent  # noqa: E402
from app.services.environmental import aiml_environmental_config  # noqa: E402
from app.services.event_upsert import process_unassigned_observations  # noqa: E402
from app.services.firms_nrt_ingestion import FirmsNRTError  # noqa: E402
from app.services.firms_observation_store import fetch_and_store_firms_nrt  # noqa: E402
from realtime.environmental import any_environmental_source_present  # noqa: E402
from src.environmental_context.config import DEFAULT_CONFIG  # noqa: E402
from src.environmental_context.raster_loader import resolve_existing_path  # noqa: E402


def _counts(session):
    return {
        "thermal_events": session.scalar(select(func.count()).select_from(ThermalEvent)),
        "active_events": session.scalar(
            select(func.count()).where(ThermalEvent.is_active.is_(True))
        ),
        "historical_events": session.scalar(
            select(func.count()).where(ThermalEvent.is_active.is_(False))
        ),
        "facilities": session.scalar(select(func.count()).select_from(Facility)),
        "firms_observations": session.scalar(
            select(func.count()).select_from(FirmsObservation)
        ),
        "event_detections": session.scalar(
            select(func.count()).select_from(EventDetection)
        ),
        "event_facility_candidates": session.scalar(
            select(func.count()).select_from(EventFacilityCandidate)
        ),
        "active_with_env_landcover_flag": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.landcover_available.is_not(None),
            )
        ),
        "active_landcover_true": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.landcover_available.is_(True),
            )
        ),
        "active_water_true": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.water_context_available.is_(True),
            )
        ),
    }


def _env_summary(session, event_id: str) -> dict:
    ev = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if ev is None:
        return {}
    return {
        "event_id": ev.event_id,
        "facility_id": ev.facility_id,
        "anomaly_status": ev.anomaly_status,
        "anomaly_score": ev.anomaly_score,
        "sta_association_status": ev.sta_association_status,
        "sta_evidence_available": ev.sta_evidence_available,
        "landcover_available": ev.landcover_available,
        "dominant_landcover_class": ev.dominant_landcover_class,
        "vegetation_context_available": ev.vegetation_context_available,
        "builtup_context_available": ev.builtup_context_available,
        "water_context_available": ev.water_context_available,
        "water_present": ev.water_present,
        "agriculture_context_available": ev.agriculture_context_available,
        "satellite_context_available": ev.satellite_context_available,
        "satellite_value": ev.satellite_value,
    }


def _source_status() -> dict:
    cfg = aiml_environmental_config(DEFAULT_CONFIG)
    paths = {
        "landcover_raster": cfg.landcover_raster_path,
        "landcover_vector": cfg.landcover_vector_path,
        "vegetation": cfg.vegetation_path,
        "builtup": cfg.builtup_path,
        "water": cfg.water_path,
        "agriculture": cfg.agriculture_path,
        "satellite_raster": cfg.satellite_raster_path,
    }
    present = {
        name: str(p) if resolve_existing_path(p) is not None else None
        for name, p in paths.items()
    }
    return {
        "any_source_present": any_environmental_source_present(cfg),
        "resolved_existing_paths": {k: v for k, v in present.items() if v is not None},
        "missing_sources": [k for k, v in present.items() if v is None],
        "note": (
            "If no local datasets: realtime writes availability=false and null "
            "evidence fields (does not fabricate environmental values)."
        ),
    }


def _run_pass(label: str) -> dict:
    session = SessionLocal()
    try:
        before = _counts(session)
        hist = session.scalar(
            select(ThermalEvent)
            .where(ThermalEvent.is_active.is_(False))
            .order_by(ThermalEvent.id.asc())
            .limit(1)
        )
        hist_before = None
        if hist is not None:
            hist_before = (
                hist.event_id,
                hist.anomaly_score,
                hist.anomaly_status,
                hist.sta_association_status,
                hist.landcover_available,
                hist.dominant_landcover_class,
                hist.water_present,
                hist.detection_count,
            )

        try:
            _df, store_stats = fetch_and_store_firms_nrt(session, commit=False)
        except FirmsNRTError as exc:
            session.rollback()
            return {"label": label, "error": str(exc), "before": before}

        recovered = session.execute(
            text(
                """
                UPDATE firms_observations o
                SET event_id = NULL
                WHERE o.event_id IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM thermal_events e WHERE e.event_id = o.event_id
                  )
                """
            )
        ).rowcount

        form_stats = process_unassigned_observations(session, commit=False)
        after = _counts(session)

        samples = []
        for eid in sorted(set(form_stats.event_ids_touched))[:20]:
            samples.append(_env_summary(session, eid))

        hist_after = None
        if hist is not None:
            session.refresh(hist)
            hist_after = (
                hist.event_id,
                hist.anomaly_score,
                hist.anomaly_status,
                hist.sta_association_status,
                hist.landcover_available,
                hist.dominant_landcover_class,
                hist.water_present,
                hist.detection_count,
            )

        session.commit()
        return {
            "label": label,
            "before": before,
            "after": after,
            "store_inserted": getattr(store_stats, "inserted", None),
            "store_duplicates": getattr(store_stats, "duplicates", None),
            "orphaned_event_ids_cleared": recovered,
            "formation": {
                "processed": form_stats.processed,
                "created": form_stats.created,
                "matched": form_stats.matched,
                "anomaly_updated": form_stats.anomaly_updated,
                "sta_updated": form_stats.sta_updated,
                "environmental_updated": form_stats.environmental_updated,
            },
            "env_samples": samples[:8],
            "historical_guard_unchanged": hist_before == hist_after,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> int:
    settings = get_settings()
    print(f"DATABASE_URL host check: ...@{settings.database_url.split('@')[-1]}")
    print(json.dumps(_source_status(), indent=2, default=str))
    print("=== Pass 1: NRT + Phases 3-9 ===")
    p1 = _run_pass("pass1")
    print(json.dumps(p1, indent=2, default=str))
    print("=== Pass 2: duplicate poll (idempotency) ===")
    p2 = _run_pass("pass2")
    print(json.dumps(p2, indent=2, default=str))
    print("=== Summary ===")
    print(
        json.dumps(
            {
                "events_before_after_pass1": (
                    p1.get("before", {}).get("thermal_events"),
                    p1.get("after", {}).get("thermal_events"),
                ),
                "historical_before_after_pass1": (
                    p1.get("before", {}).get("historical_events"),
                    p1.get("after", {}).get("historical_events"),
                ),
                "facilities_before_after_pass1": (
                    p1.get("before", {}).get("facilities"),
                    p1.get("after", {}).get("facilities"),
                ),
                "active_with_env_flag_after": p1.get("after", {}).get(
                    "active_with_env_landcover_flag"
                ),
                "active_landcover_true_after": p1.get("after", {}).get("active_landcover_true"),
                "pass2_created": p2.get("formation", {}).get("created"),
                "pass2_environmental_updated": p2.get("formation", {}).get(
                    "environmental_updated"
                ),
                "historical_guard_ok": p1.get("historical_guard_unchanged"),
                "real_environmental_data_available": _source_status()["any_source_present"],
            },
            indent=2,
        )
    )
    return 1 if "error" in p1 or "error" in p2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
