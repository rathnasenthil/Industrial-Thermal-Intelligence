"""
Restore Phase 3/4 NRT events after accidental test CASCADE wipe.

Clears stale event_id on firms_observations whose thermal_events row is gone,
then re-runs incremental formation + G.1 persistence.

Does NOT touch historical (is_active=false) Stage VII events.
Does NOT call run_persistence_characterization() over all events.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
AIML_ROOT = BACKEND_ROOT.parent / "aiml"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(AIML_ROOT))

from sqlalchemy import text  # noqa: E402

from app.db.database import SessionLocal  # noqa: E402
from app.services.event_upsert import process_unassigned_observations  # noqa: E402


def main() -> None:
    session = SessionLocal()
    try:
        cleared = session.execute(
            text(
                """
                UPDATE firms_observations fo
                SET event_id = NULL
                WHERE fo.event_id IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM thermal_events te
                    WHERE te.event_id = fo.event_id
                  )
                """
            )
        ).rowcount
        session.commit()
        print(f"Cleared stale event_id on {cleared} observations")

        stats = process_unassigned_observations(session, commit=True)
        print(json.dumps(stats.to_dict(), indent=2, default=str))
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
