"""Phase 10: realtime I.7 must match batch evidence-fusion semantics."""

from __future__ import annotations

import pandas as pd

from realtime.evidence_fusion import process_event_evidence_fusion
from src.evidence_fusion.config import (
    CANDIDATE_INDUSTRIAL,
    CANDIDATE_INSUFFICIENT,
    CANDIDATE_POSSIBLE_INDUSTRIAL,
)
from src.evidence_fusion.fusion_pipeline import run_evidence_fusion
from src.evidence_fusion.fusion_schema import FUSION_COLUMNS
from tests.fixtures.evidence_fusion.make_fixtures import make_fusion_events


def test_batch_realtime_parity() -> None:
    """Parity vs batch on the *same one-event frame* (incremental semantics)."""
    events = make_fusion_events()
    for eid in events["event_id"].astype(str):
        row = events.loc[events["event_id"] == eid]
        batch = run_evidence_fusion(row)
        rt = process_event_evidence_fusion(row, eid)
        brow = batch.events_df.iloc[0]
        for col in FUSION_COLUMNS:
            left = rt.values[col]
            right = brow[col]
            if isinstance(right, float) and pd.isna(right):
                assert left is None
            elif isinstance(left, bool) or isinstance(right, (bool, int)):
                assert bool(left) == bool(right)
            elif left is None:
                assert right is None or (isinstance(right, float) and pd.isna(right))
            elif isinstance(left, (int, float)) and not isinstance(left, bool):
                assert float(left) == float(right)
            else:
                assert str(left) == str(right)


def test_strong_facility_candidate() -> None:
    events = make_fusion_events()
    r = process_event_evidence_fusion(
        events.loc[events["event_id"] == "EVT_STRONG_FACILITY"], "EVT_STRONG_FACILITY"
    )
    assert r.get("source_intelligence_candidate") == CANDIDATE_INDUSTRIAL
    assert r.get("candidate_is_ground_truth") is False


def test_near_facility_possible() -> None:
    events = make_fusion_events()
    r = process_event_evidence_fusion(
        events.loc[events["event_id"] == "EVT_NEAR_FACILITY"], "EVT_NEAR_FACILITY"
    )
    assert r.get("source_intelligence_candidate") == CANDIDATE_POSSIBLE_INDUSTRIAL


def test_missing_sta_not_anti_industrial() -> None:
    events = make_fusion_events()
    r = process_event_evidence_fusion(
        events.loc[events["event_id"] == "EVT_STRONG_FACILITY"], "EVT_STRONG_FACILITY"
    )
    # Even with missing env, strong facility remains industrial candidate
    assert r.get("environmental_domain_available") is False
    assert r.get("source_intelligence_candidate") == CANDIDATE_INDUSTRIAL


def test_no_facility_insufficient() -> None:
    events = make_fusion_events()
    r = process_event_evidence_fusion(
        events.loc[events["event_id"] == "EVT_NO_FACILITY"], "EVT_NO_FACILITY"
    )
    assert r.get("source_intelligence_candidate") == CANDIDATE_INSUFFICIENT
    assert "NATURAL" not in str(r.get("source_intelligence_candidate"))


def test_idempotent() -> None:
    events = make_fusion_events()
    row = events.loc[events["event_id"] == "EVT_STRONG_FACILITY"]
    r1 = process_event_evidence_fusion(row, "EVT_STRONG_FACILITY")
    r2 = process_event_evidence_fusion(row, "EVT_STRONG_FACILITY")
    assert r1.to_dict() == r2.to_dict()


def test_current_event_only() -> None:
    events = make_fusion_events()
    only = events.loc[events["event_id"] == "EVT_AMBIGUOUS"]
    r = process_event_evidence_fusion(only, "EVT_AMBIGUOUS")
    assert r.event_id == "EVT_AMBIGUOUS"
    assert set(r.values.keys()) == set(FUSION_COLUMNS)


def test_i4_i5_not_in_output_mutation() -> None:
    events = make_fusion_events()
    before = events.loc[events["event_id"] == "EVT_STRONG_FACILITY"].iloc[0]
    batch = run_evidence_fusion(events.loc[events["event_id"] == "EVT_STRONG_FACILITY"])
    after = batch.events_df.iloc[0]
    assert after["anomaly_status"] == before["anomaly_status"]
    assert after["sta_association_status"] == before["sta_association_status"]
