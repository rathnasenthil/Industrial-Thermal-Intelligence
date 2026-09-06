"""
Manual Phase 1→2→3→4: fetch → store → event formation → G.1 persistence.

NOT a scheduler. Does not call AIML Stages I.2–VI / risk / alerts.
Does NOT call ``run_persistence_characterization()`` over all events.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_persistence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
AIML_ROOT = REPO_ROOT / "aiml"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.database import SessionLocal  # noqa: E402
from app.models.event_detection import EventDetection  # noqa: E402
from app.models.firms_observation import FirmsObservation  # noqa: E402
from app.models.thermal_event import ThermalEvent  # noqa: E402
from app.services.event_upsert import (  # noqa: E402
    process_unassigned_observations,
    refresh_event_persistence,
)
from app.services.firms_nrt_ingestion import FirmsNRTError  # noqa: E402
from app.services.firms_observation_store import fetch_and_store_firms_nrt  # noqa: E402


def _counts(session):
    return {
        "thermal_events": session.scalar(select(func.count()).select_from(ThermalEvent)),
        "active_events": session.scalar(
            select(func.count()).where(ThermalEvent.is_active.is_(True))
        ),
        "firms_observations": session.scalar(
            select(func.count()).select_from(FirmsObservation)
        ),
        "observations_with_event": session.scalar(
            select(func.count()).where(FirmsObservation.event_id.is_not(None))
        ),
        "event_detections": session.scalar(
            select(func.count()).select_from(EventDetection)
        ),
        "active_with_persistence_label": session.scalar(
            select(func.count()).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.persistence_label.is_not(None),
            )
        ),
        "historical_events": session.scalar(
            select(func.count()).where(ThermalEvent.is_active.is_(False))
        ),
    }


def _persistence_summary(session, event_id: str) -> dict:
    ev = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if ev is None:
        return {}
    return {
        "event_id": ev.event_id,
        "detection_count": ev.detection_count,
        "distinct_detection_days": ev.distinct_detection_days,
        "span_days": ev.span_days,
        "observed_duration_hours": ev.observed_duration_hours,
        "duty_cycle": ev.duty_cycle,
        "mean_gap_hours": ev.mean_gap_hours,
        "max_gap_hours": ev.max_gap_hours,
        "persistence_label": ev.persistence_label,
    }


def _backfill_active_missing_persistence(session) -> list[dict]:
    """
    One-time G.1 refresh for active events that already have detections but
    lack persistence_label (e.g. created in Phase 3 before Phase 4).

    Still per-event only — never full-batch ``run_persistence_characterization``.
    """
    rows = list(
        session.scalars(
            select(ThermalEvent).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.persistence_label.is_(None),
            )
        ).all()
    )
    results = []
    for ev in rows:
        before = _persistence_summary(session, ev.event_id)
        try:
            refresh_event_persistence(session, ev.event_id)
        except ValueError:
            continue
        after = _persistence_summary(session, ev.event_id)
        results.append({"before": before, "after": after})
    session.commit()
    return results


def _run_pass(label: str) -> dict:
    settings = get_settings()
    session = SessionLocal()
    try:
        before = _counts(session)
        hist_sample = session.scalar(
            select(ThermalEvent)
            .where(ThermalEvent.is_active.is_(False))
            .order_by(ThermalEvent.id.asc())
            .limit(1)
        )
        hist_before = None
        if hist_sample is not None:
            hist_before = (
                hist_sample.event_id,
                hist_sample.persistence_label,
                hist_sample.duty_cycle,
                hist_sample.detection_count,
            )

        df, store = fetch_and_store_firms_nrt(session, settings=settings)
        hashes = list(store.inserted_hashes)
        if hashes:
            formation = process_unassigned_observations(
                session,
                observation_hashes=hashes,
                commit=True,
            )
        else:
            formation = process_unassigned_observations(session, commit=True)

        after = _counts(session)
        hist_after = None
        if hist_sample is not None:
            session.refresh(hist_sample)
            hist_after = (
                hist_sample.event_id,
                hist_sample.persistence_label,
                hist_sample.duty_cycle,
                hist_sample.detection_count,
            )

        sample_persistence = []
        for eid in list(dict.fromkeys(formation.event_ids_touched))[:5]:
            sample_persistence.append(_persistence_summary(session, eid))
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
        "historical_sample_unchanged": hist_before == hist_after,
        "historical_sample": {"before": hist_before, "after": hist_after},
        "sample_persistence": sample_persistence,
        "note": "FIRMS_MAP_KEY is never printed",
    }


def main() -> None:
    print("Phase 4 realtime G.1 persistence manual test")
    print("=" * 60)

    session = SessionLocal()
    try:
        backfill = _backfill_active_missing_persistence(session)
        print(
            f"Backfilled G.1 for {len(backfill)} active events missing persistence_label"
        )
        if backfill:
            print("Example backfill:")
            print(json.dumps(backfill[0], indent=2, default=str))
    finally:
        session.close()

    # Before/after on one multi-detection active event if available.
    session = SessionLocal()
    try:
        multi = session.scalar(
            select(ThermalEvent)
            .where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.detection_count > 1,
            )
            .order_by(ThermalEvent.detection_count.desc())
            .limit(1)
        )
        before_example = None
        after_example = None
        if multi is not None:
            before_example = _persistence_summary(session, multi.event_id)
            refresh_event_persistence(session, multi.event_id)
            session.commit()
            after_example = _persistence_summary(session, multi.event_id)
    finally:
        session.close()

    if before_example is not None:
        print("\nIdempotent refresh (same detections -> same G.1):")
        print(json.dumps({"before": before_example, "after": after_example}, indent=2))
        assert before_example == after_example, "idempotency violated"

    pass1 = _run_pass("first_poll")
    print("\nFirst poll:")
    print(json.dumps(pass1, indent=2, default=str))

    pass2 = _run_pass("second_poll")
    print("\nSecond poll (expect duplicates / zero persistence churn):")
    print(json.dumps(pass2, indent=2, default=str))

    assert pass2["store"]["inserted"] == 0
    assert pass2["formation"]["created"] == 0
    assert pass2["formation"]["matched"] == 0
    assert pass2["historical_sample_unchanged"] is True
    print("\nOK: duplicate poll idempotent; historical sample unchanged.")
    print("Confirmed: only per-event G.1 (no full-batch persistence run).")


if __name__ == "__main__":
    main()
