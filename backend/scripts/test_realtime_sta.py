"""
Manual Phase 1-8 verification: NRT → … → I.4 → I.5 STA.

NOT a scheduler. Does not run fusion / risk / WebSockets.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_sta.py
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
from app.services.event_upsert import process_unassigned_observations  # noqa: E402
from app.services.firms_nrt_ingestion import FirmsNRTError  # noqa: E402
from app.services.firms_observation_store import fetch_and_store_firms_nrt  # noqa: E402
from app.services.sta import clear_sta_layer_cache  # noqa: E402
from src.sta_evidence.config import DEFAULT_CONFIG, STAConfig  # noqa: E402
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
        "active_with_sta_status": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.sta_association_status.is_not(None),
            )
        ),
        "active_sta_evidence_true": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.sta_evidence_available.is_(True),
            )
        ),
    }


def _sta_summary(session, event_id: str) -> dict:
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
        "sta_evidence_quality": ev.sta_evidence_quality,
        "sta_match_count": ev.sta_match_count,
        "primary_sta_id": ev.primary_sta_id,
        "sta_layer_type": ev.sta_layer_type,
        "sta_nearest_distance_km": ev.sta_nearest_distance_km,
    }


def _aiml_sta_paths() -> STAConfig:
    mask = DEFAULT_CONFIG.mask_path
    det = DEFAULT_CONFIG.detection_path
    if mask is not None and not Path(mask).is_absolute():
        mask = AIML_ROOT / mask
    if det is not None and not Path(det).is_absolute():
        det = AIML_ROOT / det
    return STAConfig(mask_path=mask, detection_path=det)


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
            samples.append(_sta_summary(session, eid))

        hist_after = None
        if hist is not None:
            session.refresh(hist)
            hist_after = (
                hist.event_id,
                hist.anomaly_score,
                hist.anomaly_status,
                hist.sta_association_status,
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
            },
            "sta_samples": samples[:8],
            "historical_guard_unchanged": hist_before == hist_after,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> int:
    clear_sta_layer_cache()
    settings = get_settings()
    cfg = _aiml_sta_paths()
    existing = resolve_existing_paths(cfg)
    print(f"DATABASE_URL host check: ...@{settings.database_url.split('@')[-1]}")
    print(
        json.dumps(
            {
                "sta_source_files_present": bool(existing),
                "sta_paths": [(str(p), layer) for p, layer in existing],
                "note": (
                    "If no STA files: realtime writes NO_STA_ASSOCIATION "
                    "(does not fabricate STA geometries)."
                ),
            },
            indent=2,
        )
    )
    print("=== Pass 1: NRT + Phases 3-8 ===")
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
                "candidates_before_after_pass1": (
                    p1.get("before", {}).get("event_facility_candidates"),
                    p1.get("after", {}).get("event_facility_candidates"),
                ),
                "active_with_sta_status_after": p1.get("after", {}).get("active_with_sta_status"),
                "active_sta_evidence_true_after": p1.get("after", {}).get(
                    "active_sta_evidence_true"
                ),
                "pass2_created": p2.get("formation", {}).get("created"),
                "historical_guard_ok": p1.get("historical_guard_unchanged"),
                "real_sta_data_available": bool(existing),
            },
            indent=2,
        )
    )
    return 1 if "error" in p1 or "error" in p2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
