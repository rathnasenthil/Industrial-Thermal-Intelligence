"""Tests for Stage I.7 evidence-fusion configuration."""

from __future__ import annotations

from src.evidence_fusion.config import (
    CANDIDATE_INDUSTRIAL,
    CANDIDATE_INSUFFICIENT,
    CANDIDATE_VALUES,
    EvidenceFusionConfig,
)


def test_config_defaults() -> None:
    cfg = EvidenceFusionConfig()
    assert cfg.infra_aggregate_weight == 2
    assert cfg.near_min_corroboration_for_possible == 2
    assert cfg.industrial_min_infrastructure == 3
    assert "thermal_events_with_environmental_context.csv" in str(cfg.events_path)


def test_candidate_vocabulary() -> None:
    assert CANDIDATE_INDUSTRIAL in CANDIDATE_VALUES
    assert CANDIDATE_INSUFFICIENT in CANDIDATE_VALUES
    blob = " ".join(CANDIDATE_VALUES).lower()
    assert "industrial_fire" not in blob
    assert "wildfire" not in blob
    assert "agricultural_fire" not in blob


def test_config_rationale_documents_semantics() -> None:
    rationale = EvidenceFusionConfig().describe_rationale()
    assert "missing" in rationale["missing_evidence"].lower()
    assert "not probability" in rationale["ordinal_scores"].lower() or "not probabilities" in rationale["ordinal_scores"].lower()
    assert "not independently validated" in rationale["candidate_semantics"].lower()


def test_config_to_dict_documents_ordinal_scale() -> None:
    payload = EvidenceFusionConfig().to_dict()
    assert payload["ordinal_family_scale"]["0"] == "no supporting evidence"
    assert payload["industrial_score_max_documented"] == 14
