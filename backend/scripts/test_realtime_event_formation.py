"""
Manual Phase 1→2→3 integration: fetch → store → incremental event formation.

NOT a scheduler. Does not call AIML Stages G.1–VI / risk / alerts.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_realtime_event_formation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.database import SessionLocal  # noqa: E402
from app.models.event_detection import EventDetection  # noqa: E402
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
        "firms_observations": session.scalar(
            select(func.count()).select_from(FirmsObservation)
        ),
        "observations_with_event": session.scalar(
            select(func.count()).where(FirmsObservation.event_id.is_not(None))
        ),
        "event_detections": session.scalar(
            select(func.count()).select_from(EventDetection)
        ),
    }


def _run_pass(label: str) -> dict:
    settings = get_settings()
    session = SessionLocal()
    try:
        before = _counts(session)
        df, store = fetch_and_store_firms_nrt(session, settings=settings)
        # Prefer newly inserted hashes; also process any remaining unassigned.
        hashes = list(store.inserted_hashes)
        if not hashes:
            # Second poll: still run formation over unassigned (should be none).
            formation = process_unassigned_observations(session, commit=True)
        else:
            formation = process_unassigned_observations(
                session,
                observation_hashes=hashes,
                commit=True,
            )
        after = _counts(session)
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
    }


def main() -> int:
    settings = get_settings()
    print("Phase 3 realtime event formation manual test")
    print(f"  product={settings.firms_product} bbox={settings.firms_bbox}")
    print(f"  map_key={'set' if settings.firms_map_key.strip() else 'MISSING'}")
    print()

    first = _run_pass("first")
    print("=== PASS 1 ===")
    print(json.dumps(first, indent=2, default=str))
    print()

    second = _run_pass("second")
    print("=== PASS 2 (expect observation duplicates + no new detections) ===")
    print(json.dumps(second, indent=2, default=str))
    print()

    print("Summary:")
    print(
        f"  pass1 store inserted={first['store']['inserted']} "
        f"formation created={first['formation']['created']} "
        f"matched={first['formation']['matched']}"
    )
    print(
        f"  pass2 store duplicates={second['store']['duplicates']} "
        f"formation processed={second['formation']['processed']}"
    )
    print(
        f"  event_detections pass1={first['counts_after']['event_detections']} "
        f"pass2={second['counts_after']['event_detections']}"
    )
    if second["counts_after"]["event_detections"] != first["counts_after"]["event_detections"]:
        print("  WARNING: event_detections count changed on second pass")
    else:
        print("  OK: second pass created no additional event_detections")

    # Historical corpus sanity
    if first["counts_before"]["thermal_events"] < 179740:
        print(
            f"  NOTE: thermal_events before pass1 = "
            f"{first['counts_before']['thermal_events']} (expected >= 179740 if Stage VI loaded)"
        )
    else:
        print(
            f"  OK: historical thermal_events preserved "
            f"({first['counts_before']['thermal_events']} before NRT event creation)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
