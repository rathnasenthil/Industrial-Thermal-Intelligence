"""
Manual Phase 1-5 verification: NRT fetch -> store -> event -> G.1 -> I.2.

NOT a scheduler. Does not run fingerprinting / anomaly / fusion / risk.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_facility_association.py
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
from app.models.firms_observation import FirmsObservation  # noqa: E402
from app.models.thermal_event import ThermalEvent  # noqa: E402
from app.services.event_upsert import process_unassigned_observations  # noqa: E402
from app.services.facility_association import refresh_event_facility_association  # noqa: E402
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
        "firms_observations": session.scalar(
            select(func.count()).select_from(FirmsObservation)
        ),
        "event_detections": session.scalar(
            select(func.count()).select_from(EventDetection)
        ),
        "event_facility_candidates": session.scalar(
            select(func.count()).select_from(EventFacilityCandidate)
        ),
        "active_with_facility": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.facility_id.is_not(None),
            )
        ),
        "active_with_assoc_method": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.facility_association_method.is_not(None),
            )
        ),
    }


def _event_assoc_summary(session, event_id: str) -> dict:
    ev = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if ev is None:
        return {}
    cands = list(
        session.scalars(
            select(EventFacilityCandidate)
            .where(EventFacilityCandidate.event_id == event_id)
            .order_by(EventFacilityCandidate.candidate_rank.asc())
        )
    )
    return {
        "event_id": ev.event_id,
        "centroid_lat": ev.centroid_latitude,
        "centroid_lon": ev.centroid_longitude,
        "facility_id": ev.facility_id,
        "facility_name": ev.facility_name,
        "facility_type": ev.facility_type,
        "facility_association_method": ev.facility_association_method,
        "facility_attribution_confidence": ev.facility_attribution_confidence,
        "facility_distance_km": ev.facility_distance_km,
        "candidate_facility_count": ev.candidate_facility_count,
        "candidates": [
            {
                "facility_id": c.facility_id,
                "spatial_relation": c.spatial_relation,
                "distance_km": c.distance_km,
                "candidate_rank": c.candidate_rank,
                "candidate_score": c.candidate_score,
            }
            for c in cands
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
                hist.facility_association_method,
                hist.facility_distance_km,
            )
        cand_before = before["event_facility_candidates"]

        df, store = fetch_and_store_firms_nrt(session, settings=get_settings())
        hashes = list(store.inserted_hashes)
        if hashes:
            formation = process_unassigned_observations(
                session, observation_hashes=hashes, commit=True
            )
        else:
            formation = process_unassigned_observations(session, commit=True)

        after = _counts(session)
        if hist is not None:
            session.refresh(hist)
            hist_after = (
                hist.event_id,
                hist.facility_id,
                hist.facility_association_method,
                hist.facility_distance_km,
            )
        else:
            hist_after = None

        samples = []
        for eid in list(dict.fromkeys(formation.event_ids_touched))[:5]:
            samples.append(_event_assoc_summary(session, eid))
        # Also show a few active events that already have association methods.
        if not samples:
            for ev in session.scalars(
                select(ThermalEvent)
                .where(
                    ThermalEvent.is_active.is_(True),
                    ThermalEvent.facility_association_method.is_not(None),
                )
                .limit(3)
            ):
                samples.append(_event_assoc_summary(session, ev.event_id))
    except FirmsNRTError as exc:
        session.rollback()
        raise SystemExit(f"FIRMS fetch error: {exc}") from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return {
        "pass": label,
        "fetched": int(len(df)),
        "store": store.to_dict(),
        "formation": formation.to_dict(),
        "counts_before": before,
        "counts_after": after,
        "candidate_table_delta": after["event_facility_candidates"] - cand_before,
        "historical_sample_unchanged": hist_before == hist_after,
        "historical_sample": {"before": hist_before, "after": hist_after},
        "association_samples": samples,
        "note": "FIRMS_MAP_KEY is never printed; association is spatial only",
    }


def main() -> None:
    print("Phase 5 realtime I.2 facility association manual test")
    print("=" * 60)

    # Backfill association for active events that lack method (created before Phase 5).
    session = SessionLocal()
    try:
        missing = list(
            session.scalars(
                select(ThermalEvent).where(
                    ThermalEvent.is_active.is_(True),
                    ThermalEvent.facility_association_method.is_(None),
                )
            )
        )
        print(f"Backfilling I.2 for {len(missing)} active events missing association")
        for ev in missing:
            refresh_event_facility_association(session, ev.event_id)
        session.commit()
        if missing:
            print(json.dumps(_event_assoc_summary(session, missing[0].event_id), indent=2, default=str))
    finally:
        session.close()

    # Idempotent refresh demo on one active event.
    session = SessionLocal()
    try:
        ev = session.scalar(
            select(ThermalEvent)
            .where(ThermalEvent.is_active.is_(True))
            .order_by(ThermalEvent.detection_count.desc())
            .limit(1)
        )
        if ev is not None:
            before = _event_assoc_summary(session, ev.event_id)
            cand_n_before = len(before.get("candidates", []))
            refresh_event_facility_association(session, ev.event_id)
            session.commit()
            after = _event_assoc_summary(session, ev.event_id)
            print("\nIdempotent refresh (same geometry -> same association):")
            print(json.dumps({"before": before, "after": after}, indent=2, default=str))
            assert before == after
            assert len(after.get("candidates", [])) == cand_n_before
    finally:
        session.close()

    pass1 = _run_pass("first_poll")
    print("\nFirst poll:")
    print(json.dumps(pass1, indent=2, default=str))

    pass2 = _run_pass("second_poll")
    print("\nSecond poll (expect duplicates / zero association churn):")
    print(json.dumps(pass2, indent=2, default=str))

    assert pass2["store"]["inserted"] == 0
    assert pass2["formation"]["created"] == 0
    assert pass2["formation"]["matched"] == 0
    assert pass2["candidate_table_delta"] == 0
    assert pass2["historical_sample_unchanged"] is True
    print("\nOK: duplicate poll idempotent; historical sample unchanged.")
    print("Confirmed: per-event I.2 only (no full-batch facility association).")


if __name__ == "__main__":
    main()
