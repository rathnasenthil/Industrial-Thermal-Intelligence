"""
Manual integration test: FIRMS NRT fetch + Phase 2 observation store.

This is NOT a scheduler. It does not run AIML or assign event_id.

Usage (from backend/):
  .\\.venv\\Scripts\\python.exe scripts\\test_firms_nrt.py

Requires:
  - FIRMS_MAP_KEY in backend/.env
  - PostgreSQL/PostGIS reachable via DATABASE_URL
  - alembic upgrade head (includes 002_realtime_observations)

Never prints the MAP_KEY.
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
from app.models.firms_observation import FirmsObservation  # noqa: E402
from app.services.firms_nrt_ingestion import (  # noqa: E402
    FirmsNRTError,
    summarize_observations,
)
from app.services.firms_observation_store import fetch_and_store_firms_nrt  # noqa: E402


def _run_pass(label: str) -> dict:
    settings = get_settings()
    session = SessionLocal()
    try:
        df, store = fetch_and_store_firms_nrt(session, settings=settings)
    except FirmsNRTError as exc:
        session.rollback()
        session.close()
        raise SystemExit(f"ERROR fetching FIRMS: {exc}") from exc
    except Exception:
        session.rollback()
        session.close()
        raise

    try:
        total_rows = session.scalar(select(func.count()).select_from(FirmsObservation))
        null_event_ids = session.scalar(
            select(func.count()).where(FirmsObservation.event_id.is_(None))
        )
    finally:
        session.close()

    summary = summarize_observations(df)
    payload = {
        "pass": label,
        "fetch_summary": summary,
        "store": store.to_dict(),
        "db_total_rows": int(total_rows or 0),
        "db_rows_with_null_event_id": int(null_event_ids or 0),
    }
    return payload


def main() -> int:
    settings = get_settings()
    print("FIRMS NRT Phase 2 manual test (fetch + store + dedup)")
    print(f"  product   : {settings.firms_product}")
    print(f"  bbox      : {settings.firms_bbox}")
    print(f"  day_range : {settings.firms_day_range}")
    print(f"  map_key   : {'set' if settings.firms_map_key.strip() else 'MISSING'}")
    print()

    first = _run_pass("first")
    print("=== PASS 1 ===")
    print(json.dumps(first, indent=2, default=str))
    print()

    second = _run_pass("second")
    print("=== PASS 2 (expect mostly duplicates) ===")
    print(json.dumps(second, indent=2, default=str))
    print()

    print("Dedup check:")
    print(f"  pass1 inserted={first['store']['inserted']} duplicates={first['store']['duplicates']}")
    print(
        f"  pass2 inserted={second['store']['inserted']} duplicates={second['store']['duplicates']}"
    )
    print(f"  db_total_rows after pass2={second['db_total_rows']}")
    if second["store"]["duplicates"] < second["store"]["received"] * 0.5:
        print(
            "  NOTE: second pass had fewer duplicates than expected — "
            "FIRMS feed may have changed between polls."
        )
    else:
        print("  OK: second pass is majority duplicates (idempotent store).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
