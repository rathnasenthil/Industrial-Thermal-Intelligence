"""Phase 11: realtime Stage VI risk must match batch prioritization semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from realtime.risk import process_event_risk
from src.risk_prioritization.config import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    RiskPrioritizationConfig,
)
from src.risk_prioritization.priority_scoring import priority_from_score
from src.risk_prioritization.risk_pipeline import run_risk_prioritization
from src.risk_prioritization.risk_schema import RISK_APPEND_COLUMNS
from tests.fixtures.risk_prioritization.make_fixtures import make_risk_events


def test_batch_realtime_parity() -> None:
    """Parity vs batch on the *same one-event frame* (incremental semantics).

    Note: Stage VI thermal severity includes a within-frame percentile term.
    Realtime passes one event, so parity is defined against batch on that
    same one-event frame — not against a multi-event offline batch run.
    """
    events = make_risk_events()
    for eid in events["event_id"].astype(str):
        row = events.loc[events["event_id"] == eid]
        batch = run_risk_prioritization(row)
        rt = process_event_risk(row, eid)
        brow = batch.events_df.iloc[0]
        assert rt.risk_score == pytest.approx(float(brow["risk_score"]), rel=1e-9, abs=1e-9)
        assert rt.investigation_priority == str(brow["investigation_priority"])
        for col in RISK_APPEND_COLUMNS:
            left = rt.values[col]
            right = brow[col]
            if isinstance(right, float) and pd.isna(right):
                assert left is None
            elif left is None:
                assert right is None or (isinstance(right, float) and pd.isna(right))
            elif isinstance(left, (int, float)) and not isinstance(left, bool):
                assert float(left) == float(right)
            else:
                assert str(left) == str(right)


def test_priority_bands() -> None:
    cfg = RiskPrioritizationConfig()
    assert priority_from_score(10.0, cfg) == PRIORITY_LOW
    assert priority_from_score(30.0, cfg) == PRIORITY_MEDIUM
    assert priority_from_score(60.0, cfg) == PRIORITY_HIGH
    assert priority_from_score(80.0, cfg) == PRIORITY_CRITICAL


def test_high_score_strong_anomaly() -> None:
    events = make_risk_events()
    r = process_event_risk(
        events.loc[events["event_id"] == "EVT_STRONG_ANOMALY"], "EVT_STRONG_ANOMALY"
    )
    assert r.risk_score is not None and r.risk_score >= 50.0
    assert r.investigation_priority in (PRIORITY_HIGH, PRIORITY_CRITICAL)


def test_low_score_quiet() -> None:
    events = make_risk_events()
    assert "EVT_QUIET" in set(events["event_id"])
    r = process_event_risk(events.loc[events["event_id"] == "EVT_QUIET"], "EVT_QUIET")
    assert r.risk_score is not None
    assert r.investigation_priority in (PRIORITY_LOW, PRIORITY_MEDIUM)


def test_missing_sta_does_not_force_low() -> None:
    events = make_risk_events()
    r = process_event_risk(
        events.loc[events["event_id"] == "EVT_STRONG_ANOMALY"], "EVT_STRONG_ANOMALY"
    )
    assert r.investigation_priority in (PRIORITY_MEDIUM, PRIORITY_HIGH, PRIORITY_CRITICAL)


def test_idempotent() -> None:
    events = make_risk_events()
    row = events.loc[events["event_id"] == "EVT_PERSISTENT_NORMAL"]
    r1 = process_event_risk(row, "EVT_PERSISTENT_NORMAL")
    r2 = process_event_risk(row, "EVT_PERSISTENT_NORMAL")
    assert r1.to_dict() == r2.to_dict()


def test_score_in_range() -> None:
    events = make_risk_events()
    for eid in events["event_id"].astype(str):
        r = process_event_risk(events.loc[events["event_id"] == eid], eid)
        assert r.risk_score is not None
        assert 0.0 <= r.risk_score <= 100.0


def test_current_event_only() -> None:
    events = make_risk_events()
    only = events.loc[events["event_id"] == "EVT_LOW_FRP_ANOMALY"]
    r = process_event_risk(only, "EVT_LOW_FRP_ANOMALY")
    assert r.event_id == "EVT_LOW_FRP_ANOMALY"
    assert set(r.values.keys()) == set(RISK_APPEND_COLUMNS)
