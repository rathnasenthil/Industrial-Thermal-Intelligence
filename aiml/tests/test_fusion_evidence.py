"""Unit tests for Stage I.7 evidence extractors and conflict logic."""

from __future__ import annotations

import pandas as pd

from src.evidence_fusion.conflicts import detect_evidence_conflicts
from src.evidence_fusion.environmental_evidence import extract_environmental_evidence
from src.evidence_fusion.infrastructure_evidence import extract_infrastructure_evidence
from src.evidence_fusion.sta_fusion_evidence import extract_sta_evidence
from src.evidence_fusion.temporal_evidence import extract_temporal_evidence
from tests.fixtures.evidence_fusion.make_fixtures import (
    make_events_without_sta_columns,
    make_fusion_events,
)


def test_temporal_and_infrastructure_extraction() -> None:
    events = make_fusion_events()
    temporal = extract_temporal_evidence(events)
    infra = extract_infrastructure_evidence(events)
    assert temporal["temporal_evidence_available"].all()
    assert list(temporal["temporal_persistence_signal"])[0] == "PERSISTENT"
    strong = infra.loc[infra["event_id"] == "EVT_STRONG_FACILITY"].iloc[0]
    assert strong["infrastructure_association_signal"] == "CONFIRMED"
    none = infra.loc[infra["event_id"] == "EVT_NO_FACILITY"].iloc[0]
    assert none["infrastructure_association_signal"] == "NONE"


def test_sta_unavailable_when_columns_missing() -> None:
    events = make_events_without_sta_columns()
    sta = extract_sta_evidence(events)
    assert not sta["sta_domain_available"].any()
    assert (sta["sta_association_signal"] == "UNAVAILABLE").all()
    # Must not coerce to NO_STA_ASSOCIATION (that would imply a processed negative).
    assert "NO_STA_ASSOCIATION" not in set(sta["sta_association_signal"])


def test_sta_no_match_is_available_domain() -> None:
    events = make_fusion_events()
    sta = extract_sta_evidence(events)
    assert sta["sta_domain_available"].all()
    row = sta.loc[sta["event_id"] == "EVT_NO_FACILITY"].iloc[0]
    assert row["sta_association_signal"] == "NO_STA_ASSOCIATION"
    assert "not_anti_industrial" in row["sta_evidence_summary"]


def test_environmental_null_semantics() -> None:
    events = make_fusion_events()
    env = extract_environmental_evidence(events)
    no_env = env.loc[env["event_id"] == "EVT_NO_FACILITY"].iloc[0]
    assert no_env["environmental_domain_available"] is False or no_env["environmental_domain_available"] == False
    assert no_env["environmental_vegetation_signal"] is None or pd.isna(no_env["environmental_vegetation_signal"])
    veg = env.loc[env["event_id"] == "EVT_VEG_CONTEXT"].iloc[0]
    assert bool(veg["environmental_domain_available"]) is True
    assert veg["environmental_vegetation_signal"] == "PRESENT"


def test_conflicts_require_available_evidence() -> None:
    events = make_fusion_events()
    infra = extract_infrastructure_evidence(events)
    sta = extract_sta_evidence(events)
    env = extract_environmental_evidence(events)
    conflicts = detect_evidence_conflicts(infra, sta, env)
    mixed = conflicts.loc[conflicts["event_id"] == "EVT_MIXED"].iloc[0]
    assert bool(mixed["evidence_conflict_flag"]) is True
    assert "FACILITY_VS_AGRICULTURE" in mixed["evidence_conflict_codes"]
    no_fac = conflicts.loc[conflicts["event_id"] == "EVT_NO_FACILITY"].iloc[0]
    assert bool(no_fac["evidence_conflict_flag"]) is False
