"""
Manual Phase 1-6 verification: NRT fetch -> store -> event -> G.1 -> I.2 -> I.3.

NOT a scheduler. Does not run anomaly / fusion / risk / WebSockets.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_facility_fingerprint.py
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
        "active_with_facility": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.facility_id.is_not(None),
            )
        ),
    }


def _fp_summary(session, facility_id: str) -> dict:
    fp = session.scalar(
        select(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == facility_id
        )
    )
    if fp is None:
        return {"facility_id": facility_id, "present": False}
    monthly = list(
        session.scalars(
            select(FacilityMonthlyThermalProfile)
            .where(FacilityMonthlyThermalProfile.facility_id == facility_id)
            .order_by(FacilityMonthlyThermalProfile.month.asc())
        )
    )
    return {
        "facility_id": facility_id,
        "present": True,
        "event_count": fp.event_count,
        "detection_count": fp.detection_count,
        "fingerprint_status": fp.fingerprint_status,
        "ambiguous_candidate_opportunity_count": fp.ambiguous_candidate_opportunity_count,
        "peak_frp_median": fp.peak_frp_median,
        "monthly_rows": [
            {
                "month": m.month,
                "event_count": m.event_count,
                "detection_count": m.detection_count,
                "event_fraction": m.event_fraction,
            }
            for m in monthly
        ],
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
                hist.facility_id,
                hist.detection_count,
                hist.anomaly_score,
            )

        # Snapshot one existing fingerprint (if any) to prove unrelated rows untouched.
        existing_fp = session.scalar(
            select(FacilityThermalFingerprint).order_by(FacilityThermalFingerprint.id.asc()).limit(1)
        )
        existing_fp_before = None
        if existing_fp is not None:
            existing_fp_before = (
                existing_fp.facility_id,
                existing_fp.event_count,
                existing_fp.fingerprint_status,
                existing_fp.updated_at,
            )

        try:
            store_stats = fetch_and_store_firms_nrt(session, commit=False)
        except FirmsNRTError as exc:
            session.rollback()
            return {"label": label, "error": str(exc), "before": before}

        form_stats = process_unassigned_observations(session, commit=False)
        after = _counts(session)

        touched = []
        for eid in sorted(set(form_stats.event_ids_touched)):
            ev = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
            if ev is None:
                continue
            entry = {
                "event_id": eid,
                "facility_id": ev.facility_id,
                "facility_association_method": ev.facility_association_method,
                "fingerprint": _fp_summary(session, ev.facility_id) if ev.facility_id else None,
            }
            touched.append(entry)

        hist_after = None
        if hist is not None:
            session.refresh(hist)
            hist_after = (
                hist.event_id,
                hist.facility_id,
                hist.detection_count,
                hist.anomaly_score,
            )

        existing_fp_after = None
        if existing_fp is not None:
            session.refresh(existing_fp)
            existing_fp_after = (
                existing_fp.facility_id,
                existing_fp.event_count,
                existing_fp.fingerprint_status,
                existing_fp.updated_at,
            )

        session.commit()
        return {
            "label": label,
            "before": before,
            "after": after,
            "store": store_stats.to_dict() if hasattr(store_stats, "to_dict") else store_stats,
            "formation": form_stats.to_dict(),
            "touched_events": touched,
            "historical_guard_unchanged": hist_before == hist_after,
            "existing_fingerprint_unchanged_if_unrelated": (
                existing_fp_before == existing_fp_after
                if existing_fp_before and existing_fp_before[0]
                not in {
                    t["facility_id"]
                    for t in touched
                    if t.get("facility_id")
                }
                else None
            ),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> int:
    settings = get_settings()
    print(f"DATABASE_URL host check: ...@{settings.database_url.split('@')[-1]}")
    print("=== Pass 1: NRT + Phases 3-6 ===")
    p1 = _run_pass("pass1")
    print(json.dumps(p1, indent=2, default=str))
    print("=== Pass 2: duplicate poll (idempotency) ===")
    p2 = _run_pass("pass2")
    print(json.dumps(p2, indent=2, default=str))

    if "error" in p1 or "error" in p2:
        return 1
    # Duplicate poll should not create new events or drift fingerprint counts for same facilities.
    print("=== Summary ===")
    print(
        json.dumps(
            {
                "events_before_after_pass1": (p1["before"]["thermal_events"], p1["after"]["thermal_events"]),
                "events_before_after_pass2": (p2["before"]["thermal_events"], p2["after"]["thermal_events"]),
                "facilities_before_after_pass1": (p1["before"]["facilities"], p1["after"]["facilities"]),
                "candidates_before_after_pass1": (
                    p1["before"]["event_facility_candidates"],
                    p1["after"]["event_facility_candidates"],
                ),
                "fingerprints_before_after_pass1": (
                    p1["before"]["facility_thermal_fingerprints"],
                    p1["after"]["facility_thermal_fingerprints"],
                ),
                "monthly_before_after_pass1": (
                    p1["before"]["facility_monthly_thermal_profile"],
                    p1["after"]["facility_monthly_thermal_profile"],
                ),
                "pass2_created": p2["formation"]["created"],
                "historical_guard_ok": p1.get("historical_guard_unchanged"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
