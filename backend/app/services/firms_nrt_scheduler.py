"""
Phase 12: lightweight APScheduler wrapper for FIRMS NRT polling.

Default disabled (``FIRMS_NRT_ENABLED=false``) so tests/dev do not hit NASA.
Does not start on import. Does not start under pytest.
Uses existing FIRMS NRT fetch/store + ``process_unassigned_observations``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.database import SessionLocal
from app.services.event_upsert import process_unassigned_observations
from app.services.firms_nrt_ingestion import FirmsNRTError
from app.services.firms_observation_store import fetch_and_store_firms_nrt
from sqlalchemy import text

logger = logging.getLogger(__name__)

_scheduler = None
_poll_lock = threading.Lock()
_poll_running = False


def _pytest_running() -> bool:
    return (
        "pytest" in sys.modules
        or os.environ.get("PYTEST_CURRENT_TEST") is not None
        or os.environ.get("FIRMS_NRT_FORCE_DISABLE") == "1"
    )


def run_firms_nrt_poll_cycle(
    session: Optional[Session] = None,
    *,
    settings: Optional[Settings] = None,
    commit: bool = True,
) -> dict[str, Any]:
    """
    One full realtime poll: FIRMS fetch → store → Phases 3–11.

    Safe to call manually from scripts/tests. Overlap guard is optional
    (scheduler uses ``_locked_poll``).
    """
    own_session = session is None
    db = session or SessionLocal()
    cfg = settings or get_settings()
    try:
        try:
            _df, store_stats = fetch_and_store_firms_nrt(db, settings=cfg, commit=False)
        except FirmsNRTError as exc:
            db.rollback()
            logger.error("FIRMS NRT poll fetch failed: %s", exc)
            return {"ok": False, "error": str(exc), "stage": "fetch"}

        # Recover orphaned observation→event links (deleted thermal events).
        recovered = db.execute(
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

        form_stats = process_unassigned_observations(db, commit=False)
        if commit:
            db.commit()
        else:
            db.flush()

        payload = {
            "ok": True,
            "store_inserted": getattr(store_stats, "inserted", None),
            "store_duplicates": getattr(store_stats, "duplicates", None),
            "orphaned_event_ids_cleared": recovered,
            "formation": form_stats.to_dict(),
        }
        logger.info(
            "FIRMS NRT poll complete: inserted=%s duplicates=%s processed=%s "
            "created=%s matched=%s fusion=%s risk=%s",
            payload["store_inserted"],
            payload["store_duplicates"],
            form_stats.processed,
            form_stats.created,
            form_stats.matched,
            form_stats.fusion_updated,
            form_stats.risk_updated,
        )
        return payload
    except Exception:
        db.rollback()
        logger.exception("FIRMS NRT poll cycle failed")
        raise
    finally:
        if own_session:
            db.close()


def _locked_poll() -> None:
    """Scheduler job: skip if previous poll still running; never kill the scheduler."""
    global _poll_running
    if not _poll_lock.acquire(blocking=False):
        logger.warning("FIRMS NRT poll skipped: previous poll still running")
        return
    try:
        if _poll_running:
            logger.warning("FIRMS NRT poll skipped: overlap guard")
            return
        _poll_running = True
        try:
            run_firms_nrt_poll_cycle(commit=True)
        except Exception as exc:
            logger.error("FIRMS NRT scheduled poll error (scheduler continues): %s", exc)
        finally:
            _poll_running = False
    finally:
        _poll_lock.release()


def start_firms_nrt_scheduler(settings: Optional[Settings] = None) -> bool:
    """
    Start BackgroundScheduler when enabled and not under pytest.

    Returns True if scheduler started, False if skipped.
    """
    global _scheduler
    cfg = settings or get_settings()

    if _pytest_running():
        logger.info("FIRMS NRT scheduler not started (pytest detected)")
        return False
    if not cfg.firms_nrt_enabled:
        logger.info("FIRMS NRT scheduler disabled (FIRMS_NRT_ENABLED=false)")
        return False
    if _scheduler is not None:
        logger.info("FIRMS NRT scheduler already running")
        return True

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError as exc:
        logger.error("APScheduler not installed; cannot start FIRMS NRT scheduler: %s", exc)
        return False

    interval = max(1, int(cfg.firms_nrt_interval_minutes))
    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        _locked_poll,
        trigger=IntervalTrigger(minutes=interval),
        id="firms_nrt_poll",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    sched.start()
    _scheduler = sched
    logger.info(
        "FIRMS NRT scheduler started: interval=%s minutes product=%s",
        interval,
        cfg.firms_product,
    )
    return True


def stop_firms_nrt_scheduler() -> None:
    """Shut down scheduler cleanly (idempotent)."""
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
        logger.info("FIRMS NRT scheduler stopped")
    except Exception:
        logger.exception("Error stopping FIRMS NRT scheduler")
    finally:
        _scheduler = None


def is_firms_nrt_scheduler_running() -> bool:
    return _scheduler is not None and getattr(_scheduler, "running", False)
