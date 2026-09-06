"""
Manual Phase 1-7 verification: NRT → event → G.1 → I.2 → I.3 → I.4.

NOT a scheduler. Does not run fusion / risk / WebSockets.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_anomaly.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
AIML_ROOT = BACKEND_ROOT.parent / "aiml"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(AIML_ROOT))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.database import SessionLocal  # noqa: E402
from app.models.event_detection import EventDetection  # noqa: E402
from app.models.event_facility_candidate import EventFacilityCandidate  # noqa: E402
from app.models.facility import Facility  # noqa: E402
from app.models.facility_monthly_thermal_profile import (  # noqa: E402
    FacilityMonthlyThermalProfile,
)
from app.models.facility_thermal_fingerprint import FacilityThermalFingerprint  # noqa: E402
from app.models.firms_observation import FirmsObservation  # noqa: E402
from app.models.thermal_event import ThermalEvent  # noqa: E402
from app.services.event_upsert import process_unassigned_observations  # noqa: E402
from app.services.firms_nrt_ingestion import FirmsNRTError  # noqa: E402
from app.services.firms_observation_store import fetch_and_store_firms_nrt  # noqa: E402


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
        "facility_thermal_fingerprints": session.scalar(
            select(func.count()).select_from(FacilityThermalFingerprint)
        ),
        "facility_monthly_thermal_profile": session.scalar(
            select(func.count()).select_from(FacilityMonthlyThermalProfile)
        ),
        "active_with_anomaly_status": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.anomaly_status.is_not(None),
            )
        ),
    }


def _anomaly_summary(session, event_id: str) -> dict:
    ev = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if ev is None:
        return {}
    return {
        "event_id": ev.event_id,
        "facility_id": ev.facility_id,
        "facility_association_method": ev.facility_association_method,
        "anomaly_score": ev.anomaly_score,
        "anomaly_status": ev.anomaly_status,
        "anomaly_confidence": ev.anomaly_confidence,
        "baseline_observation_count": ev.baseline_observation_count,
        "baseline_history_status": ev.baseline_history_status,
        "anomaly_unavailable_reason": ev.anomaly_unavailable_reason,
        "peak_frp_deviation": ev.peak_frp_deviation,
        "features_evaluated": ev.features_evaluated,
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
                hist.detection_count,
            )

        try:
            _df, store_stats = fetch_and_store_firms_nrt(session, commit=False)
        except FirmsNRTError as exc:
            session.rollback()
            return {"label": label, "error": str(exc), "before": before}

        # Recover observations whose event_id points at a deleted thermal_event
        # (leftover from earlier test cleanups). Does not touch valid linkages.
        from sqlalchemy import text

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
        for eid in sorted(set(form_stats.event_ids_touched))[:15]:
            samples.append(_anomaly_summary(session, eid))
        # Prefer samples that have a facility / non-null score when available.
        scored = [s for s in samples if s.get("anomaly_score") is not None]
        unavailable = [s for s in samples if s.get("anomaly_unavailable_reason")]

        hist_after = None
        if hist is not None:
            session.refresh(hist)
            hist_after = (
                hist.event_id,
                hist.anomaly_score,
                hist.anomaly_status,
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
            },
            "anomaly_samples_scored": scored[:5],
            "anomaly_samples_unavailable": unavailable[:5],
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
    print("=== Pass 1: NRT + Phases 3-7 ===")
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
                "events_before_after_pass2": (
                    p2.get("before", {}).get("thermal_events"),
                    p2.get("after", {}).get("thermal_events"),
                ),
                "historical_before_after_pass1": (
                    p1.get("before", {}).get("historical_events"),
                    p1.get("after", {}).get("historical_events"),
                ),
                "facilities_before_after_pass1": (
                    p1.get("before", {}).get("facilities"),
                    p1.get("after", {}).get("facilities"),
                ),
                "candidates_before_after_pass1": (
                    p1.get("before", {}).get("event_facility_candidates"),
                    p1.get("after", {}).get("event_facility_candidates"),
                ),
                "pass2_created": p2.get("formation", {}).get("created"),
                "historical_guard_ok": p1.get("historical_guard_unchanged"),
            },
            indent=2,
        )
    )
    return 1 if "error" in p1 or "error" in p2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
