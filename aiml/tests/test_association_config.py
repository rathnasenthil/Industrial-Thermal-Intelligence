"""Tests for `src.infrastructure.association_config`."""

from __future__ import annotations

from pathlib import Path

from src.infrastructure.association_config import AssociationConfig


def test_defaults_are_documented_engineering_values() -> None:
    config = AssociationConfig()
    assert config.association_radius_km == 5.0
    assert config.ambiguity_distance_tolerance_km == 0.5
    assert config.max_candidates_per_event == 10


def test_to_dict_is_json_serializable_and_paths_are_strings() -> None:
    config = AssociationConfig()
    d = config.to_dict()
    assert isinstance(d["events_path"], str)
    assert isinstance(d["facilities_path"], str)
    assert d["association_radius_km"] == 5.0


def test_describe_rationale_covers_every_threshold() -> None:
    config = AssociationConfig()
    rationale = config.describe_rationale()
    for key in ("association_radius_km", "ambiguity_distance_tolerance_km", "max_candidates_per_event"):
        assert key in rationale
        assert isinstance(rationale[key], str)
        assert len(rationale[key]) > 20


def test_config_is_configurable_not_hardcoded() -> None:
    config = AssociationConfig(
        association_radius_km=2.5,
        ambiguity_distance_tolerance_km=0.1,
        max_candidates_per_event=3,
        events_path=Path("custom_events.csv"),
        facilities_path=Path("custom_facilities.geojson"),
    )
    assert config.association_radius_km == 2.5
    assert config.ambiguity_distance_tolerance_km == 0.1
    assert config.max_candidates_per_event == 3
    assert config.to_dict()["events_path"] == "custom_events.csv"


def test_max_candidates_per_event_can_be_disabled() -> None:
    config = AssociationConfig(max_candidates_per_event=None)
    assert config.max_candidates_per_event is None
    assert config.to_dict()["max_candidates_per_event"] is None
