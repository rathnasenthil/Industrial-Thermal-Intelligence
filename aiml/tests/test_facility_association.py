"""Tests for `src.infrastructure.facility_association`."""

from __future__ import annotations

import pandas as pd
import pytest

from src.infrastructure.association_config import AssociationConfig
from src.infrastructure.association_geometry import INTERSECTS_FACILITY, NEAR_FACILITY, WITHIN_FACILITY
from src.infrastructure.facility_association import (
    AMBIGUOUS,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NONE,
    MAIN_OUTPUT_COLUMNS,
    NO_FACILITY_ASSOCIATION,
    rank_candidates,
    select_association,
)


def _pairs(rows: list[dict]) -> pd.DataFrame:
    defaults = {"facility_name": "Name", "facility_type": "OTHER_INDUSTRIAL", "geometry_type": "Point"}
    return pd.DataFrame([{**defaults, **r} for r in rows])


# --------------------------------------------------------------------------
# rank_candidates
# --------------------------------------------------------------------------


def test_rank_candidates_empty_input() -> None:
    empty = pd.DataFrame(
        columns=["event_id", "facility_id", "facility_name", "facility_type", "geometry_type", "distance_km", "spatial_relation"]
    )
    ranked = rank_candidates(empty)
    assert len(ranked) == 0
    assert "candidate_rank" in ranked.columns
    assert "candidate_score" in ranked.columns


def test_rank_candidates_relation_tier_dominates_distance() -> None:
    # A far NEAR candidate must still rank below a close WITHIN candidate.
    pairs = _pairs(
        [
            {"event_id": "E1", "facility_id": "F_within", "distance_km": 4.9, "spatial_relation": WITHIN_FACILITY},
            {"event_id": "E1", "facility_id": "F_near", "distance_km": 0.01, "spatial_relation": NEAR_FACILITY},
        ]
    )
    ranked = rank_candidates(pairs)
    top = ranked.loc[ranked["candidate_rank"] == 1].iloc[0]
    assert top["facility_id"] == "F_within"


def test_rank_candidates_distance_breaks_ties_within_tier() -> None:
    pairs = _pairs(
        [
            {"event_id": "E1", "facility_id": "F_far", "distance_km": 3.0, "spatial_relation": NEAR_FACILITY},
            {"event_id": "E1", "facility_id": "F_close", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY},
        ]
    )
    ranked = rank_candidates(pairs)
    top = ranked.loc[ranked["candidate_rank"] == 1].iloc[0]
    assert top["facility_id"] == "F_close"


def test_rank_candidates_deterministic_tie_break_by_geometry_then_type_then_id() -> None:
    # Identical relation tier and identical distance -- must be resolved
    # deterministically by geometry quality (Polygon > Point), then
    # facility_type (lexical), then facility_id (lexical).
    pairs = _pairs(
        [
            {"event_id": "E1", "facility_id": "F_point", "distance_km": 2.0, "spatial_relation": NEAR_FACILITY, "geometry_type": "Point"},
            {"event_id": "E1", "facility_id": "F_polygon", "distance_km": 2.0, "spatial_relation": NEAR_FACILITY, "geometry_type": "Polygon"},
        ]
    )
    ranked = rank_candidates(pairs)
    top = ranked.loc[ranked["candidate_rank"] == 1].iloc[0]
    assert top["facility_id"] == "F_polygon"


def test_rank_candidates_final_tiebreak_is_facility_id() -> None:
    pairs = _pairs(
        [
            {"event_id": "E1", "facility_id": "F_zzz", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY, "geometry_type": "Point", "facility_type": "MINE"},
            {"event_id": "E1", "facility_id": "F_aaa", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY, "geometry_type": "Point", "facility_type": "MINE"},
        ]
    )
    ranked = rank_candidates(pairs)
    top = ranked.loc[ranked["candidate_rank"] == 1].iloc[0]
    assert top["facility_id"] == "F_aaa"


def test_rank_candidates_is_order_independent() -> None:
    rows = [
        {"event_id": "E1", "facility_id": "F_b", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY},
        {"event_id": "E1", "facility_id": "F_a", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY},
    ]
    ranked_forward = rank_candidates(_pairs(rows))
    ranked_reversed = rank_candidates(_pairs(list(reversed(rows))))
    pd.testing.assert_frame_equal(
        ranked_forward.sort_values("facility_id").reset_index(drop=True),
        ranked_reversed.sort_values("facility_id").reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# select_association
# --------------------------------------------------------------------------


def test_select_association_single_within_is_high_confidence() -> None:
    pairs = rank_candidates(_pairs([{"event_id": "E1", "facility_id": "F1", "distance_km": 0.0, "spatial_relation": WITHIN_FACILITY}]))
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig())
    row = result.iloc[0]
    assert row["facility_id"] == "F1"
    assert row["facility_association_method"] == WITHIN_FACILITY
    assert row["facility_attribution_confidence"] == CONFIDENCE_HIGH
    assert row["candidate_facility_count"] == 1


def test_select_association_single_near_is_medium_confidence() -> None:
    pairs = rank_candidates(_pairs([{"event_id": "E1", "facility_id": "F1", "distance_km": 2.0, "spatial_relation": NEAR_FACILITY}]))
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig())
    row = result.iloc[0]
    assert row["facility_association_method"] == NEAR_FACILITY
    assert row["facility_attribution_confidence"] == CONFIDENCE_MEDIUM


def test_select_association_multiple_distinguishable_near_is_low_confidence_but_resolved() -> None:
    pairs = rank_candidates(
        _pairs(
            [
                {"event_id": "E1", "facility_id": "F_close", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY},
                {"event_id": "E1", "facility_id": "F_far", "distance_km": 4.0, "spatial_relation": NEAR_FACILITY},
            ]
        )
    )
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig(ambiguity_distance_tolerance_km=0.5))
    row = result.iloc[0]
    # Clearly resolvable (3 km apart, well above the 0.5 km tolerance) but
    # multiple candidates existed -> LOW, not MEDIUM/HIGH.
    assert row["facility_id"] == "F_close"
    assert row["facility_association_method"] == NEAR_FACILITY
    assert row["facility_attribution_confidence"] == CONFIDENCE_LOW
    assert row["candidate_facility_count"] == 2


def test_select_association_marks_ambiguous_when_candidates_too_close() -> None:
    pairs = rank_candidates(
        _pairs(
            [
                {"event_id": "E1", "facility_id": "F_a", "distance_km": 2.00, "spatial_relation": NEAR_FACILITY},
                {"event_id": "E1", "facility_id": "F_b", "distance_km": 2.05, "spatial_relation": NEAR_FACILITY},
            ]
        )
    )
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig(ambiguity_distance_tolerance_km=0.5))
    row = result.iloc[0]
    assert row["facility_association_method"] == AMBIGUOUS
    assert row["facility_attribution_confidence"] == CONFIDENCE_LOW
    # Do-not-blindly-select-nearest rule: no single facility_id chosen.
    assert pd.isna(row["facility_id"])
    assert row["candidate_facility_count"] == 2
    assert row["candidate_facility_ids"] == "F_a,F_b"


def test_select_association_different_tier_never_ambiguous_even_if_distance_close() -> None:
    # WITHIN (distance 0) vs. NEAR at a small distance -- different tiers,
    # so the WITHIN candidate must win outright, never AMBIGUOUS.
    pairs = rank_candidates(
        _pairs(
            [
                {"event_id": "E1", "facility_id": "F_within", "distance_km": 0.0, "spatial_relation": WITHIN_FACILITY},
                {"event_id": "E1", "facility_id": "F_near", "distance_km": 0.1, "spatial_relation": NEAR_FACILITY},
            ]
        )
    )
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig(ambiguity_distance_tolerance_km=0.5))
    row = result.iloc[0]
    assert row["facility_association_method"] == WITHIN_FACILITY
    assert row["facility_id"] == "F_within"


def test_select_association_no_candidates_is_none() -> None:
    empty_pairs = rank_candidates(
        pd.DataFrame(columns=["event_id", "facility_id", "facility_name", "facility_type", "geometry_type", "distance_km", "spatial_relation"])
    )
    result = select_association(pd.Series(["E1"]), empty_pairs, AssociationConfig())
    row = result.iloc[0]
    assert row["facility_association_method"] == NO_FACILITY_ASSOCIATION
    assert row["facility_attribution_confidence"] == CONFIDENCE_NONE
    assert pd.isna(row["facility_id"])
    assert row["candidate_facility_count"] == 0


def test_select_association_never_drops_an_event_without_any_candidates() -> None:
    pairs = rank_candidates(_pairs([{"event_id": "E1", "facility_id": "F1", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY}]))
    all_event_ids = pd.Series(["E1", "E2", "E3"])  # E2/E3 have no candidate rows at all
    result = select_association(all_event_ids, pairs, AssociationConfig())
    assert len(result) == 3
    assert set(result["event_id"]) == {"E1", "E2", "E3"}
    for eid in ("E2", "E3"):
        row = result.loc[result["event_id"] == eid].iloc[0]
        assert row["facility_association_method"] == NO_FACILITY_ASSOCIATION


def test_select_association_output_has_expected_columns() -> None:
    pairs = rank_candidates(_pairs([{"event_id": "E1", "facility_id": "F1", "distance_km": 1.0, "spatial_relation": NEAR_FACILITY}]))
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig())
    for col in MAIN_OUTPUT_COLUMNS:
        assert col in result.columns


def test_select_association_is_deterministic_across_repeated_calls() -> None:
    pairs = rank_candidates(
        _pairs(
            [
                {"event_id": "E1", "facility_id": "F_a", "distance_km": 2.00, "spatial_relation": NEAR_FACILITY},
                {"event_id": "E1", "facility_id": "F_b", "distance_km": 2.05, "spatial_relation": NEAR_FACILITY},
                {"event_id": "E2", "facility_id": "F_c", "distance_km": 0.0, "spatial_relation": WITHIN_FACILITY},
            ]
        )
    )
    all_event_ids = pd.Series(["E1", "E2"])
    r1 = select_association(all_event_ids, pairs, AssociationConfig())
    r2 = select_association(all_event_ids, pairs, AssociationConfig())
    pd.testing.assert_frame_equal(r1, r2)


def test_intersects_facility_is_also_high_confidence() -> None:
    pairs = rank_candidates(_pairs([{"event_id": "E1", "facility_id": "F1", "distance_km": 0.3, "spatial_relation": INTERSECTS_FACILITY}]))
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig())
    row = result.iloc[0]
    assert row["facility_association_method"] == INTERSECTS_FACILITY
    assert row["facility_attribution_confidence"] == CONFIDENCE_HIGH


@pytest.mark.parametrize("distance_km", [0.0, 5.0, 4.999999])
def test_distance_km_is_never_negative_in_selection(distance_km: float) -> None:
    pairs = rank_candidates(_pairs([{"event_id": "E1", "facility_id": "F1", "distance_km": distance_km, "spatial_relation": NEAR_FACILITY}]))
    result = select_association(pd.Series(["E1"]), pairs, AssociationConfig())
    assert result.iloc[0]["facility_distance_km"] >= 0
