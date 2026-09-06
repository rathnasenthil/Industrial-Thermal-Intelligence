"""CLI entrypoint for Stage VI / I.1 / I.2 CSV ingestion."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow `python scripts/ingest_stage_vi.py` from backend/
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.database import SessionLocal  # noqa: E402
from app.services.ingestion import (  # noqa: E402
    DEFAULT_CANDIDATE_CSV,
    DEFAULT_EVENT_CSV,
    DEFAULT_FACILITY_CSV,
    run_ingestion,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-load frozen AIML Stage VI events, I.1 facilities, and I.2 "
            "candidates into PostgreSQL/PostGIS. Does not modify aiml/."
        )
    )
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENT_CSV)
    parser.add_argument("--facilities-csv", type=Path, default=DEFAULT_FACILITY_CSV)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument(
        "--mode",
        choices=["replace"],
        default="replace",
        help="replace = truncate tables then reload (idempotent)",
    )
    parser.add_argument(
        "--skip-candidates",
        action="store_true",
        help="Skip event_facility_candidates load",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write ingestion report JSON",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        report = run_ingestion(
            session,
            events_csv=args.events_csv,
            facilities_csv=args.facilities_csv,
            candidates_csv=args.candidates_csv,
            mode=args.mode,
            load_candidates=not args.skip_candidates,
        )
    except Exception:
        session.rollback()
        logger.exception("Ingestion failed")
        return 1
    finally:
        session.close()

    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if args.report_json:
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if report.errors:
        logger.error("Ingestion completed with integrity errors: %s", report.errors)
        return 2

    logger.info(
        "Ingestion OK: events=%s facilities=%s candidates=%s",
        report.events_inserted,
        report.facilities_inserted,
        report.candidates_inserted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
