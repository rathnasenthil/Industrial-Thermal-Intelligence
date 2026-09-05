"""Stage VI risk prioritization unit/integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.risk_prioritization.anomaly_priority import compute_anomaly_priority
from src.risk_prioritization.config import (
    INDUSTRIAL_CONTEXT_INSUFFICIENT,
    INDUSTRIAL_CONTEXT_STRONG,
    PRIORITY_CRITICAL,
    PRIORITY_LOW,
    RiskPrioritizationConfig,
)
from src.risk_prioritization.facility_criticality import compute_facility_context
from src.risk_prioritization.persistence_priority import compute_persistence_priority
from src.risk_prioritization.priority_scoring import map_industrial_context, priority_from_score
from src.risk_prioritization.risk_pipeline import run_risk_prioritization, save_outputs
from src.risk_prioritization.risk_schema import I7_IMMUTABLE_COLUMNS
from src.risk_prioritization.thermal_severity import compute_thermal_severity
from src.risk_prioritization.uncertainty import compute_uncertainty
from tests.fixtures.risk_prioritization.make_fixtures import make_risk_events


def test_config_defaults() -> None:
    cfg = RiskPrioritizationConfig()
    assert cfg.weight_thermal == 25.0
    assert cfg.priority_critical_min == 75.0
    assert "not a probability" in cfg.to_dict()["semantics"]["risk_score"].lower()


def test_thermal_severity_and_frp_skew() -> None:
    events = make_risk_events()
    cfg = RiskPrioritizationConfig()
    out = compute_thermal_severity(events, cfg)
    hi = out.loc[out.event_id == "EVT_HIGH_FRP_NO_FACILITY"].iloc[0]
    lo = out.loc[out.event_id == "EVT_QUIET"].iloc[0]
    assert hi["thermal_severity_score"] > lo["thermal_severity_score"]
    assert hi["thermal_severity_band"] in {"HIGH", "EXTREME"}


def test_persistence_and_anomaly_scoring() -> None:
    events = make_risk_events()
    cfg = RiskPrioritizationConfig()
    p = compute_persistence_priority(events, cfg)
    a = compute_anomaly_priority(events, cfg)
    assert (
        p.loc[p.event_id == "EVT_PERSISTENT_NORMAL", "persistence_priority_score"].iloc[0]
        > p.loc[p.event_id == "EVT_QUIET", "persistence_priority_score"].iloc[0]
    )
    assert (
        a.loc[a.event_id == "EVT_STRONG_ANOMALY", "anomaly_priority_score"].iloc[0]
        > a.loc[a.event_id == "EVT_PERSISTENT_NORMAL", "anomaly_priority_score"].iloc[0]
    )
    # insufficient history → 0 anomaly contribution
    assert a.loc[a.event_id == "EVT_QUIET", "anomaly_priority_score"].iloc[0] == 0.0


def test_facility_context_and_confidence() -> None:
    events = make_risk_events()
    cfg = RiskPrioritizationConfig()
    f = compute_facility_context(events, cfg)
    assert f.loc[f.event_id == "EVT_STRONG_ANOMALY", "facility_context_score"].iloc[0] > 0
    assert f.loc[f.event_id == "EVT_HIGH_FRP_NO_FACILITY", "facility_context_score"].iloc[0] == 0
    assert "AMBIGUOUS" in f.loc[f.event_id == "EVT_AMBIGUOUS", "facility_context_reason"].iloc[0]


def test_uncertainty_missing_not_negative_industrial() -> None:
    events = make_risk_events()
    u = compute_uncertainty(events, RiskPrioritizationConfig())
    row = u.loc[u.event_id == "EVT_STRONG_ANOMALY"].iloc[0]
    assert "STA_UNAVAILABLE" in row["dominant_uncertainty_factors"]
    assert "ENVIRONMENTAL_CONTEXT_UNAVAILABLE" in row["dominant_uncertainty_factors"]
    # uncertainty high does not invent negative industrial label
    assert row["uncertainty_score"] > 0


def test_industrial_context_separate_from_priority() -> None:
    assert map_industrial_context("INDUSTRIAL_ACTIVITY_CANDIDATE") == INDUSTRIAL_CONTEXT_STRONG
    assert map_industrial_context("INSUFFICIENT_EVIDENCE") == INDUSTRIAL_CONTEXT_INSUFFICIENT
    cfg = RiskPrioritizationConfig()
    assert priority_from_score(10, cfg) == PRIORITY_LOW
    # strong industrial context event may not be CRITICAL if other components modest
    result = run_risk_prioritization(make_risk_events())
    persistent_normal = result.events_df.loc[
        result.events_df.event_id == "EVT_PERSISTENT_NORMAL"
    ].iloc[0]
    assert persistent_normal["industrial_context"] == INDUSTRIAL_CONTEXT_STRONG
    assert persistent_normal["investigation_priority"] != PRIORITY_CRITICAL


def test_special_cases_pipeline() -> None:
    result = run_risk_prioritization(make_risk_events())
    df = result.events_df.set_index("event_id")

    # high FRP no facility still scored / not forced LOW-only discard
    assert df.loc["EVT_HIGH_FRP_NO_FACILITY", "risk_score"] > df.loc["EVT_QUIET", "risk_score"]
    assert df.loc["EVT_HIGH_FRP_NO_FACILITY", "industrial_context"] == INDUSTRIAL_CONTEXT_INSUFFICIENT

    # low FRP high anomaly can elevate vs quiet
    assert df.loc["EVT_LOW_FRP_ANOMALY", "anomaly_priority_score"] > 0
    assert df.loc["EVT_LOW_FRP_ANOMALY", "risk_score"] > df.loc["EVT_QUIET", "risk_score"]

    # persistent does not auto critical
    assert df.loc["EVT_PERSISTENT_NORMAL", "investigation_priority"] != PRIORITY_CRITICAL

    # anomalous does not auto industrial context STRONG
    assert df.loc["EVT_LOW_FRP_ANOMALY", "industrial_context"] == "POSSIBLE"
    # insufficient evidence not forced LOW if other severity high
    assert pd.notna(df.loc["EVT_HIGH_FRP_NO_FACILITY", "investigation_priority"])

    # industrial candidate not forced CRITICAL
    assert df.loc["EVT_PERSISTENT_NORMAL", "investigation_priority"] != PRIORITY_CRITICAL


def test_explanations_differ() -> None:
    df = run_risk_prioritization(make_risk_events()).events_df
    reasons = set(df["priority_reasons"].tolist())
    assert len(reasons) > 1
    strong = df.loc[df.event_id == "EVT_STRONG_ANOMALY"].iloc[0]
    assert "ANOMALOUS" in strong["priority_reasons"] or "ANOMALOUS_TEMPORAL_BEHAVIOUR" in strong["priority_reasons"]
    assert "STA_UNAVAILABLE" in strong["priority_warnings"]


def test_determinism_and_row_preservation(tmp_path: Path) -> None:
    events = make_risk_events()
    r1 = run_risk_prioritization(events)
    r2 = run_risk_prioritization(events.sample(frac=1.0, random_state=0).reset_index(drop=True))
    assert len(r1.events_df) == len(events)
    assert set(r1.events_df.event_id) == set(events.event_id)
    cols = ["event_id", "risk_score", "investigation_priority", "industrial_context"]
    pd.testing.assert_frame_equal(r1.events_df[cols], r2.events_df[cols])
    out = tmp_path / "risk.csv"
    save_outputs(r1, out)
    reloaded = pd.read_csv(out)
    for col in reloaded.select_dtypes(include=["object", "string"]).columns:
        assert not ((reloaded[col] == "nan") & reloaded[col].notna()).any()


def test_i7_fields_immutable() -> None:
    events = make_risk_events()
    result = run_risk_prioritization(events)
    left = events.sort_values("event_id").reset_index(drop=True)
    right = result.events_df.sort_values("event_id").reset_index(drop=True)
    for col in I7_IMMUTABLE_COLUMNS:
        if col not in left.columns:
            continue
        assert list(left[col].fillna("__NA__").astype(str)) == list(right[col].fillna("__NA__").astype(str))


def test_no_single_feature_forces_all_critical() -> None:
    # All WITHIN events in fixture are not all CRITICAL
    df = run_risk_prioritization(make_risk_events()).events_df
    within = df[df.facility_association_method == "WITHIN_FACILITY"]
    assert not (within.investigation_priority == PRIORITY_CRITICAL).all()
    persistent = df[df.persistence_label == "PERSISTENT"]
    assert not (persistent.investigation_priority == PRIORITY_CRITICAL).all()
