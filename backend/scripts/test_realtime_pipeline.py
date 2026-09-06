"""
End-to-end realtime pipeline verification (Phases 1–12 orchestration path).

FIRMS NRT → store → G → G.1 → I.2 → I.3 → I.4 → I.5 → I.6 → I.7 → Risk

NOT a scheduler. Does not fabricate STA/environmental datasets.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_pipeline.py
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
from app.models.firms_observation import FirmsObservation  # noqa: E402
from app.models.thermal_event import ThermalEvent  # noqa: E402
from app.services.environmental import aiml_environmental_config  # noqa: E402
from app.services.firms_nrt_scheduler import run_firms_nrt_poll_cycle  # noqa: E402
from app.services.sta import _aiml_sta_config  # noqa: E402
from realtime.environmental import any_environmental_source_present  # noqa: E402
from src.environmental_context.config import DEFAULT_CONFIG as ENV_DEFAULT  # noqa: E402
from src.sta_evidence.config import DEFAULT_CONFIG as STA_DEFAULT  # noqa: E402
from src.sta_evidence.sta_loader import resolve_existing_paths  # noqa: E402


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
        "active_with_risk": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.risk_score.is_not(None),
            )
        ),
        "active_with_fusion": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.source_intelligence_candidate.is_not(None),
            )
        ),
    }


def _event_snapshot(session, event_id: str) -> dict:
    ev = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if ev is None:
        return {}
    return {
        "event_id": ev.event_id,
        "facility_id": ev.facility_id,
        "facility_association_method": ev.facility_association_method,
        "persistence_label": ev.persistence_label,
        "anomaly_status": ev.anomaly_status,
        "anomaly_score": ev.anomaly_score,
        "sta_association_status": ev.sta_association_status,
        "sta_evidence_available": ev.sta_evidence_available,
        "landcover_available": ev.landcover_available,
        "water_context_available": ev.water_context_available,
        "source_intelligence_candidate": ev.source_intelligence_candidate,
        "evidence_strength": ev.evidence_strength,
        "evidence_fusion_score": ev.evidence_fusion_score,
        "candidate_is_ground_truth": ev.candidate_is_ground_truth,
        "risk_score": ev.risk_score,
        "investigation_priority": ev.investigation_priority,
        "industrial_context": ev.industrial_context,
        "recommended_action": ev.recommended_action,
    }


def _source_status() -> dict:
    sta_cfg = _aiml_sta_config(STA_DEFAULT)
    env_cfg = aiml_environmental_config(ENV_DEFAULT)
    return {
        "sta_files_present": bool(resolve_existing_paths(sta_cfg)),
        "environmental_sources_present": any_environmental_source_present(env_cfg),
        "note": "Missing STA/env sources use unavailable semantics (no fabrication).",
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
                hist.risk_score,
                hist.investigation_priority,
                hist.source_intelligence_candidate,
                hist.anomaly_score,
                hist.sta_association_status,
                hist.landcover_available,
                hist.detection_count,
            )

        result = run_firms_nrt_poll_cycle(session, commit=False)
        if not result.get("ok"):
            session.rollback()
            return {"label": label, "error": result.get("error"), "before": before}

        after = _counts(session)
        form = result.get("formation") or {}
        touched = form.get("event_ids_touched") or []
        samples = [_event_snapshot(session, eid) for eid in sorted(set(touched))[:12]]

        hist_after = None
        if hist is not None:
            session.refresh(hist)
            hist_after = (
                hist.event_id,
                hist.risk_score,
                hist.investigation_priority,
                hist.source_intelligence_candidate,
                hist.anomaly_score,
                hist.sta_association_status,
                hist.landcover_available,
                hist.detection_count,
            )

        session.commit()
        return {
            "label": label,
            "before": before,
            "after": after,
            "poll": {
                "store_inserted": result.get("store_inserted"),
                "store_duplicates": result.get("store_duplicates"),
                "orphaned_event_ids_cleared": result.get("orphaned_event_ids_cleared"),
            },
            "formation": {
                "processed": form.get("processed"),
                "created": form.get("created"),
                "matched": form.get("matched"),
                "anomaly_updated": form.get("anomaly_updated"),
                "sta_updated": form.get("sta_updated"),
                "environmental_updated": form.get("environmental_updated"),
                "fusion_updated": form.get("fusion_updated"),
                "risk_updated": form.get("risk_updated"),
            },
            "samples": samples[:8],
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
    print(json.dumps(_source_status(), indent=2))
    print("=== Pass 1: full realtime chain ===")
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
                "facilities_unchanged": (
                    p1.get("before", {}).get("facilities")
                    == p1.get("after", {}).get("facilities")
                ),
                "active_with_fusion_after": p1.get("after", {}).get("active_with_fusion"),
                "active_with_risk_after": p1.get("after", {}).get("active_with_risk"),
                "pass2_created": p2.get("formation", {}).get("created"),
                "pass2_processed": p2.get("formation", {}).get("processed"),
                "pass2_store_inserted": p2.get("poll", {}).get("store_inserted"),
                "historical_guard_ok": p1.get("historical_guard_unchanged"),
                "example_event": (p1.get("samples") or [None])[0],
            },
            indent=2,
            default=str,
        )
    )
    return 1 if "error" in p1 or "error" in p2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
