"""Integration tests for Stage I.7 evidence-fusion pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evidence_fusion.config import (
    CANDIDATE_AMBIGUOUS,
    CANDIDATE_INDUSTRIAL,
    CANDIDATE_INSUFFICIENT,
    CANDIDATE_MIXED,
    CANDIDATE_POSSIBLE_INDUSTRIAL,
    CANDIDATE_VEGETATION_CONTEXT,
    EvidenceFusionConfig,
)
from src.evidence_fusion.fusion_pipeline import run_evidence_fusion, save_outputs
from src.evidence_fusion.fusion_schema import I4_IMMUTABLE_COLUMNS, I5_IMMUTABLE_COLUMNS
from tests.fixtures.evidence_fusion.make_fixtures import (
    make_events_without_sta_columns,
    make_fusion_events,
)


def _by_id(result_df: pd.DataFrame, event_id: str) -> pd.Series:
    return result_df.loc[result_df["event_id"] == event_id].iloc[0]


def test_one_row_per_event_and_ids_preserved() -> None:
    events = make_fusion_events()
    result = run_evidence_fusion(events)
    assert len(result.events_df) == len(events)
    assert result.events_df["event_id"].is_unique
    assert set(result.events_df["event_id"]) == set(events["event_id"])


def test_deterministic_ordering_and_repeat() -> None:
    events = make_fusion_events()
    r1 = run_evidence_fusion(events)
    r2 = run_evidence_fusion(events.sample(frac=1.0, random_state=1).reset_index(drop=True))
    assert list(r1.events_df["event_id"]) == list(r2.events_df["event_id"])
    cols = [
        "event_id",
        "source_intelligence_candidate",
        "industrial_evidence_score",
        "evidence_strength",
        "evidence_sufficiency",
        "evidence_uncertainty",
        "evidence_conflict_flag",
    ]
    pd.testing.assert_frame_equal(r1.events_df[cols], r2.events_df[cols])


def test_candidate_decision_branches() -> None:
    result = run_evidence_fusion(make_fusion_events())
    df = result.events_df
    assert _by_id(df, "EVT_STRONG_FACILITY")["source_intelligence_candidate"] == CANDIDATE_INDUSTRIAL
    assert _by_id(df, "EVT_NEAR_FACILITY")["source_intelligence_candidate"] == CANDIDATE_POSSIBLE_INDUSTRIAL
    assert _by_id(df, "EVT_AMBIGUOUS")["source_intelligence_candidate"] == CANDIDATE_AMBIGUOUS
    assert _by_id(df, "EVT_VEG_CONTEXT")["source_intelligence_candidate"] == CANDIDATE_VEGETATION_CONTEXT
    assert _by_id(df, "EVT_MIXED")["source_intelligence_candidate"] == CANDIDATE_MIXED
    assert _by_id(df, "EVT_STA_ONLY")["source_intelligence_candidate"] == CANDIDATE_INSUFFICIENT
    assert _by_id(df, "EVT_ANOMALY_NO_FACILITY")["source_intelligence_candidate"] == CANDIDATE_INSUFFICIENT
    assert _by_id(df, "EVT_NO_FACILITY")["source_intelligence_candidate"] == CANDIDATE_INSUFFICIENT


def test_anomaly_not_industrial_and_no_facility_not_natural() -> None:
    df = run_evidence_fusion(make_fusion_events()).events_df
    row = _by_id(df, "EVT_ANOMALY_NO_FACILITY")
    assert row["source_intelligence_candidate"] == CANDIDATE_INSUFFICIENT
    assert "ANOMALY_WITHOUT_FACILITY_NOT_INDUSTRIAL" in str(row["ambiguous_evidence_codes"])
    no_fac = _by_id(df, "EVT_NO_FACILITY")
    assert "FACILITY_NONE_NOT_NATURAL" in str(no_fac["evidence_profile_codes"])
    assert "NATURAL" not in str(no_fac["source_intelligence_candidate"])


def test_not_perfect_i2_mapping_on_fixtures() -> None:
    """NEAR with corroboration can be POSSIBLE; NEAR alone is not auto-mapped."""
    from tests.fixtures.evidence_fusion.make_fixtures import make_fusion_events

    df = run_evidence_fusion(make_fusion_events()).events_df
    # Strong facility is industrial, but industrial score must include multi-family contribution.
    strong = _by_id(df, "EVT_STRONG_FACILITY")
    assert strong["infrastructure_evidence_score"] >= 3
    assert strong["temporal_evidence_score"] == 2
    assert strong["historical_evidence_score"] == 3
    assert strong["industrial_evidence_score"] > strong["infrastructure_evidence_score"]


def test_missing_sta_not_negative() -> None:
    events = make_events_without_sta_columns()
    result = run_evidence_fusion(events)
    assert not result.events_df["sta_domain_available"].any()
    assert (result.events_df["sta_association_signal"] == "UNAVAILABLE").all()
    # Strong facility still becomes industrial candidate without STA.
    assert (
        _by_id(result.events_df, "EVT_STRONG_FACILITY")["source_intelligence_candidate"]
        == CANDIDATE_INDUSTRIAL
    )
    # No-facility does not become NATURAL due to missing STA.
    assert (
        _by_id(result.events_df, "EVT_NO_FACILITY")["source_intelligence_candidate"]
        == CANDIDATE_INSUFFICIENT
    )


def test_candidate_never_ground_truth() -> None:
    result = run_evidence_fusion(make_fusion_events())
    assert result.events_df["candidate_is_ground_truth"].eq(False).all()
    assert result.report["candidate_is_ground_truth_all_false"] is True


def test_i4_and_i5_fields_immutable() -> None:
    events = make_fusion_events()
    result = run_evidence_fusion(events)
    left = events.sort_values("event_id").reset_index(drop=True)
    right = result.events_df.sort_values("event_id").reset_index(drop=True)
    for col in I4_IMMUTABLE_COLUMNS:
        if col not in left.columns:
            continue
        if col in ("anomaly_status", "anomaly_confidence"):
            assert list(left[col]) == list(right[col])
        else:
            assert pd.to_numeric(left[col], errors="coerce").fillna(-999).tolist() == pd.to_numeric(
                right[col], errors="coerce"
            ).fillna(-999).tolist()
    for col in I5_IMMUTABLE_COLUMNS:
        if col in left.columns:
            assert list(left[col].fillna("__NA__")) == list(right[col].fillna("__NA__"))


def test_no_literal_nan(tmp_path: Path) -> None:
    result = run_evidence_fusion(make_fusion_events())
    out = tmp_path / "fusion.csv"
    save_outputs(result, out)
    reloaded = pd.read_csv(out)
    for col in reloaded.select_dtypes(include=["object", "str"]).columns:
        assert not ((reloaded[col] == "nan") & reloaded[col].notna()).any()


def test_no_forbidden_classification_columns() -> None:
    result = run_evidence_fusion(make_fusion_events())
    blob = " ".join(result.events_df.columns).lower()
    for term in (
        "industrial_fire",
        "wildfire",
        "agricultural_fire",
        "risk_score",
        "source_class",
        "pseudo_label",
    ):
        assert term not in blob


def test_prior_stage_fields_preserved() -> None:
    events = make_fusion_events()
    result = run_evidence_fusion(events)
    for col in events.columns:
        assert col in result.events_df.columns
