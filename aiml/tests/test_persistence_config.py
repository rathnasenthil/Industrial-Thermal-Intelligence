"""Tests for src.persistence.config (PersistenceConfig defaults/rationale)."""

from __future__ import annotations

from src.persistence.config import PersistenceConfig


def test_default_thresholds_are_documented_engineering_values() -> None:
    config = PersistenceConfig()
    assert config.min_detections_for_classification == 3
    assert config.short_lived_max_duration_hours == 48.0
    assert config.persistent_min_duty_cycle == 0.85
    assert config.persistent_max_gap_hours == 24.0


def test_config_is_immutable() -> None:
    config = PersistenceConfig()
    try:
        config.min_detections_for_classification = 10  # type: ignore[misc]
        assert False, "expected FrozenInstanceError"
    except Exception:
        pass


def test_to_dict_round_trips_all_fields() -> None:
    config = PersistenceConfig(min_detections_for_classification=5)
    d = config.to_dict()
    assert d["min_detections_for_classification"] == 5
    assert set(d.keys()) == {
        "min_detections_for_classification",
        "short_lived_max_duration_hours",
        "persistent_min_duty_cycle",
        "persistent_max_gap_hours",
    }


def test_describe_rationale_covers_every_field() -> None:
    config = PersistenceConfig()
    rationale = config.describe_rationale()
    assert set(rationale.keys()) == set(config.to_dict().keys())
    for text in rationale.values():
        assert isinstance(text, str) and len(text) > 0
