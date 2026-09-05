"""Tests proving multi-family scoring (not I.2-only mapping)."""

from __future__ import annotations

import pandas as pd

from src.evidence_fusion.config import (
    CANDIDATE_AMBIGUOUS,
    CANDIDATE_INDUSTRIAL,
    CANDIDATE_INSUFFICIENT,
    CANDIDATE_MIXED,
    CANDIDATE_POSSIBLE_INDUSTRIAL,
)
from src.evidence_fusion.evidence_scores import (
    score_anomaly,
    score_historical,
    score_infrastructure,
    score_temporal,
)
from src.evidence_fusion.fusion_pipeline import run_evidence_fusion
from tests.fixtures.evidence_fusion.make_fixtures import make_fusion_events


def _base(**overrides) -> pd.DataFrame:
    row = {
        "event_id": "EVT_X",
        "persistence_label": "SHORT_LIVED",
        "anomaly_status": "NORMAL",
        "anomaly_score": 0.0,
        "anomaly_confidence": "MEDIUM",
        "peak_frp_deviation": 0.0,
        "event_size_deviation": 0.0,
        "duration_deviation": 0.0,
        "distance_deviation": 0.0,
        "persistence_deviation": 0.0,
        "monthly_deviation": 0.0,
        "facility_association_method": "WITHIN_FACILITY",
        "facility_attribution_confidence": "HIGH",
        "facility_type": "POWER_PLANT",
        "baseline_history_status": "NO_PRIOR_OBSERVATIONS",
        "sta_association_status": "NO_STA_ASSOCIATION",
        "sta_evidence_quality": "NONE",
        "sta_layer_type": None,
        "landcover_available": False,
        "vegetation_context_available": False,
        "builtup_context_available": False,
        "water_context_available": False,
        "agriculture_context_available": False,
        "satellite_context_available": False,
        "vegetation_present": None,
        "agriculture_present": None,
        "builtup_present": None,
        "water_present": None,
        "dominant_landcover_class": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_same_association_different_persistence_changes_score() -> None:
    a = run_evidence_fusion(_base(event_id="A", persistence_label="SHORT_LIVED")).events_df.iloc[0]
    b = run_evidence_fusion(_base(event_id="B", persistence_label="PERSISTENT")).events_df.iloc[0]
    assert a["facility_association_method"] == b["facility_association_method"]
    assert a["temporal_evidence_score"] != b["temporal_evidence_score"]
    assert b["industrial_evidence_score"] > a["industrial_evidence_score"]


def test_same_association_different_anomaly_changes_score() -> None:
    a = run_evidence_fusion(_base(event_id="A", anomaly_status="NORMAL")).events_df.iloc[0]
    b = run_evidence_fusion(_base(event_id="B", anomaly_status="ANOMALOUS")).events_df.iloc[0]
    assert a["anomaly_evidence_score"] != b["anomaly_evidence_score"]
    assert b["industrial_evidence_score"] > a["industrial_evidence_score"]
    # Candidate may remain INDUSTRIAL (strong infra), but strength/score differs.
    assert a["source_intelligence_candidate"] == CANDIDATE_INDUSTRIAL
    assert b["source_intelligence_candidate"] == CANDIDATE_INDUSTRIAL
    assert a["industrial_evidence_score"] != b["industrial_evidence_score"]


def test_same_association_different_history_changes_score() -> None:
    a = run_evidence_fusion(
        _base(event_id="A", baseline_history_status="NO_PRIOR_OBSERVATIONS")
    ).events_df.iloc[0]
    b = run_evidence_fusion(
        _base(event_id="B", baseline_history_status="ESTABLISHED_BASELINE")
    ).events_df.iloc[0]
    assert a["historical_evidence_score"] != b["historical_evidence_score"]
    assert b["industrial_evidence_score"] > a["industrial_evidence_score"]


def test_persistent_no_facility_not_industrial() -> None:
    row = run_evidence_fusion(
        _base(
            event_id="P",
            facility_association_method="NO_FACILITY_ASSOCIATION",
            facility_attribution_confidence="NONE",
            facility_type=None,
            baseline_history_status="NOT_APPLICABLE",
            persistence_label="PERSISTENT",
        )
    ).events_df.iloc[0]
    assert row["source_intelligence_candidate"] == CANDIDATE_INSUFFICIENT
    assert row["temporal_evidence_score"] == 2
    assert row["infrastructure_evidence_score"] == 0


def test_anomalous_no_facility_not_industrial() -> None:
    row = run_evidence_fusion(
        _base(
            event_id="A",
            facility_association_method="NO_FACILITY_ASSOCIATION",
            facility_attribution_confidence="NONE",
            facility_type=None,
            baseline_history_status="NOT_APPLICABLE",
            anomaly_status="ANOMALOUS",
            anomaly_confidence="HIGH",
        )
    ).events_df.iloc[0]
    assert row["source_intelligence_candidate"] == CANDIDATE_INSUFFICIENT
    assert row["anomaly_evidence_score"] == 2
    assert "ANOMALY_WITHOUT_FACILITY_NOT_INDUSTRIAL" in str(row["ambiguous_evidence_codes"])


def test_near_alone_not_strongest_or_auto_possible() -> None:
    row = run_evidence_fusion(
        _base(
            event_id="N",
            facility_association_method="NEAR_FACILITY",
            facility_attribution_confidence="LOW",
            facility_type="INDUSTRIAL_AREA",
            persistence_label="SHORT_LIVED",
            anomaly_status="NORMAL",
            baseline_history_status="NO_PRIOR_OBSERVATIONS",
        )
    ).events_df.iloc[0]
    assert row["source_intelligence_candidate"] == CANDIDATE_INSUFFICIENT
    assert row["source_intelligence_candidate"] != CANDIDATE_INDUSTRIAL
    assert row["infrastructure_evidence_score"] < 3


def test_within_with_corroboration_stronger_than_within_alone() -> None:
    alone = run_evidence_fusion(
        _base(
            event_id="A",
            persistence_label="SHORT_LIVED",
            anomaly_status="NORMAL",
            baseline_history_status="NO_PRIOR_OBSERVATIONS",
        )
    ).events_df.iloc[0]
    rich = run_evidence_fusion(
        _base(
            event_id="B",
            persistence_label="PERSISTENT",
            anomaly_status="NORMAL",
            baseline_history_status="ESTABLISHED_BASELINE",
        )
    ).events_df.iloc[0]
    assert rich["industrial_evidence_score"] > alone["industrial_evidence_score"]
    assert alone["source_intelligence_candidate"] == CANDIDATE_INDUSTRIAL
    assert rich["source_intelligence_candidate"] == CANDIDATE_INDUSTRIAL


def test_missing_sta_not_negative() -> None:
    with_sta = make_fusion_events().iloc[[0]].copy()
    without = with_sta.drop(columns=["sta_association_status", "sta_evidence_quality", "sta_layer_type"])
    r1 = run_evidence_fusion(with_sta).events_df.iloc[0]
    r2 = run_evidence_fusion(without).events_df.iloc[0]
    # Removing STA association should not invent a penalty beyond losing optional +2 max.
    assert r2["sta_evidence_score"] == 0
    assert r2["source_intelligence_candidate"] == CANDIDATE_INDUSTRIAL
    assert "STA_UNAVAILABLE" in str(r2["limiting_evidence_codes"])


def test_missing_env_not_natural_evidence() -> None:
    row = run_evidence_fusion(_base(event_id="E")).events_df.iloc[0]
    assert row["environmental_evidence_score"] == 0
    assert row["environmental_support_score"] == 0
    assert "ENVIRONMENTAL_CONTEXT_UNAVAILABLE" in str(row["limiting_evidence_codes"])
    assert row["source_intelligence_candidate"] != "NATURAL"


def test_ambiguous_remains_ambiguous() -> None:
    row = run_evidence_fusion(
        _base(
            event_id="AMB",
            facility_association_method="AMBIGUOUS",
            facility_attribution_confidence="LOW",
            facility_type=None,
            baseline_history_status="NOT_APPLICABLE",
            persistence_label="PERSISTENT",
            anomaly_status="ANOMALOUS",
        )
    ).events_df.iloc[0]
    assert row["source_intelligence_candidate"] == CANDIDATE_AMBIGUOUS


def test_industrial_plus_environmental_mixed(tmp_path=None) -> None:
    row = run_evidence_fusion(
        _base(
            event_id="M",
            persistence_label="PERSISTENT",
            baseline_history_status="ESTABLISHED_BASELINE",
            vegetation_context_available=True,
            agriculture_context_available=True,
            vegetation_present=True,
            agriculture_present=True,
        )
    ).events_df.iloc[0]
    assert row["source_intelligence_candidate"] == CANDIDATE_MIXED
    assert row["environmental_support_score"] >= 2


def test_no_weak_family_alone_makes_industrial() -> None:
    for overrides in (
        {"facility_association_method": "NO_FACILITY_ASSOCIATION", "facility_attribution_confidence": "NONE",
         "facility_type": None, "baseline_history_status": "NOT_APPLICABLE", "persistence_label": "PERSISTENT"},
        {"facility_association_method": "NO_FACILITY_ASSOCIATION", "facility_attribution_confidence": "NONE",
         "facility_type": None, "baseline_history_status": "NOT_APPLICABLE", "anomaly_status": "ANOMALOUS"},
        {"facility_association_method": "NEAR_FACILITY", "facility_attribution_confidence": "LOW",
         "facility_type": "UNKNOWN", "persistence_label": "SHORT_LIVED", "anomaly_status": "NORMAL",
         "baseline_history_status": "NO_PRIOR_OBSERVATIONS"},
    ):
        row = run_evidence_fusion(_base(event_id="W", **overrides)).events_df.iloc[0]
        assert row["source_intelligence_candidate"] != CANDIDATE_INDUSTRIAL


def test_near_with_corroboration_can_be_possible() -> None:
    row = run_evidence_fusion(
        _base(
            event_id="NC",
            facility_association_method="NEAR_FACILITY",
            facility_attribution_confidence="MEDIUM",
            facility_type="MINE",
            persistence_label="PERSISTENT",
            anomaly_status="ELEVATED",
            baseline_history_status="LIMITED_HISTORY",
        )
    ).events_df.iloc[0]
    assert row["source_intelligence_candidate"] == CANDIDATE_POSSIBLE_INDUSTRIAL
    assert row["infrastructure_evidence_score"] < 3


def test_ordinal_score_helpers() -> None:
    assert score_infrastructure("WITHIN_FACILITY", "HIGH", "POWER_PLANT") == 3
    assert score_infrastructure("NEAR_FACILITY", "LOW", "INDUSTRIAL_AREA") == 1
    assert score_temporal("PERSISTENT") == 2
    assert score_temporal("SHORT_LIVED") == 0
    assert score_historical("ESTABLISHED_BASELINE", "WITHIN_FACILITY") == 3
    assert score_historical("ESTABLISHED_BASELINE", "NO_FACILITY_ASSOCIATION") == 0
    assert score_anomaly("ANOMALOUS") == 2
    assert score_anomaly("NORMAL") == 0


def test_explanations_match_contributing_evidence() -> None:
    row = run_evidence_fusion(
        _base(
            event_id="EX",
            persistence_label="PERSISTENT",
            baseline_history_status="ESTABLISHED_BASELINE",
            anomaly_status="ANOMALOUS",
        )
    ).events_df.iloc[0]
    supporting = str(row["supporting_evidence_codes"])
    assert "FACILITY_WITHIN_FACILITY" in supporting
    assert "TEMPORAL_PERSISTENT" in supporting
    assert "HISTORICAL_ESTABLISHED_BASELINE" in supporting
    assert "TEMPORAL_ANOMALOUS_DEVIATION" in supporting
    assert "ENVIRONMENTAL_CONTEXT_UNAVAILABLE" in str(row["limiting_evidence_codes"])
    # STA domain present with no match → limiting NO_STA_MATCH (not a negative industrial score).
    assert "NO_STA_MATCH" in str(row["limiting_evidence_codes"]) or "STA_UNAVAILABLE" in str(
        row["limiting_evidence_codes"]
    )
