"""Tests for Phase 12 FIRMS NRT scheduler lifecycle (no NASA calls)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services import firms_nrt_scheduler as sched


@pytest.fixture(autouse=True)
def _reset_scheduler():
    sched.stop_firms_nrt_scheduler()
    yield
    sched.stop_firms_nrt_scheduler()


def test_scheduler_disabled_by_default() -> None:
    settings = SimpleNamespace(firms_nrt_enabled=False, firms_nrt_interval_minutes=15, firms_product="X")
    with patch.object(sched, "_pytest_running", return_value=False):
        assert sched.start_firms_nrt_scheduler(settings) is False
    assert sched.is_firms_nrt_scheduler_running() is False


def test_scheduler_not_started_under_pytest() -> None:
    settings = SimpleNamespace(firms_nrt_enabled=True, firms_nrt_interval_minutes=15, firms_product="X")
    # Real pytest environment → must refuse to start
    assert sched.start_firms_nrt_scheduler(settings) is False


def test_scheduler_starts_when_enabled_and_not_pytest() -> None:
    pytest.importorskip("apscheduler")
    settings = SimpleNamespace(
        firms_nrt_enabled=True,
        firms_nrt_interval_minutes=15,
        firms_product="VIIRS_NOAA20_NRT",
    )
    with patch.object(sched, "_pytest_running", return_value=False):
        assert sched.start_firms_nrt_scheduler(settings) is True
        assert sched.is_firms_nrt_scheduler_running() is True
        # Idempotent second start
        assert sched.start_firms_nrt_scheduler(settings) is True
        sched.stop_firms_nrt_scheduler()
    assert sched.is_firms_nrt_scheduler_running() is False


def test_overlap_guard_skips_second_poll() -> None:
    calls = {"n": 0}

    def slow_cycle(**_kwargs):
        calls["n"] += 1
        return {"ok": True}

    with patch.object(sched, "run_firms_nrt_poll_cycle", side_effect=slow_cycle):
        # Simulate overlap: set running flag then call locked poll
        sched._poll_running = True
        try:
            sched._locked_poll()
            assert calls["n"] == 0
        finally:
            sched._poll_running = False


def test_poll_error_does_not_raise_from_locked_job() -> None:
    with patch.object(sched, "run_firms_nrt_poll_cycle", side_effect=RuntimeError("boom")):
        # Must not raise — scheduler job swallows and logs
        sched._locked_poll()


def test_run_cycle_uses_existing_services() -> None:
    fake_store = SimpleNamespace(inserted=0, duplicates=3)
    fake_form = MagicMock()
    fake_form.to_dict.return_value = {"processed": 0}
    fake_form.processed = 0
    fake_form.created = 0
    fake_form.matched = 0
    fake_form.fusion_updated = 0
    fake_form.risk_updated = 0

    session = MagicMock()
    session.execute.return_value.rowcount = 0

    with (
        patch.object(sched, "fetch_and_store_firms_nrt", return_value=(MagicMock(), fake_store)),
        patch.object(sched, "process_unassigned_observations", return_value=fake_form),
    ):
        out = sched.run_firms_nrt_poll_cycle(session, commit=False)
    assert out["ok"] is True
    assert out["store_duplicates"] == 3
    session.commit.assert_not_called()
    session.flush.assert_called()
