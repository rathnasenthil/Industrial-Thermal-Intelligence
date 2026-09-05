"""Tests for `src.fingerprinting.fingerprint_config`."""

from __future__ import annotations

import pytest

from src.fingerprinting.fingerprint_config import (
    ESTABLISHED_BASELINE,
    FingerprintConfig,
    INSUFFICIENT_HISTORY,
    LIMITED_HISTORY,
    NO_OBSERVATIONS,
)


def test_defaults() -> None:
    config = FingerprintConfig()
    assert config.min_observations_for_limited_history == 3
    assert config.min_observations_for_established_baseline == 10


@pytest.mark.parametrize(
    ("event_count", "expected_status"),
    [
        (0, NO_OBSERVATIONS),
        (1, INSUFFICIENT_HISTORY),
        (2, INSUFFICIENT_HISTORY),
        (3, LIMITED_HISTORY),
        (9, LIMITED_HISTORY),
        (10, ESTABLISHED_BASELINE),
        (500, ESTABLISHED_BASELINE),
    ],
)
def test_classify_status_default_thresholds(event_count: int, expected_status: str) -> None:
    config = FingerprintConfig()
    assert config.classify_status(event_count) == expected_status


def test_classify_status_is_configurable() -> None:
    config = FingerprintConfig(min_observations_for_limited_history=5, min_observations_for_established_baseline=20)
    assert config.classify_status(4) == INSUFFICIENT_HISTORY
    assert config.classify_status(5) == LIMITED_HISTORY
    assert config.classify_status(19) == LIMITED_HISTORY
    assert config.classify_status(20) == ESTABLISHED_BASELINE


def test_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        FingerprintConfig(min_observations_for_limited_history=0)
    with pytest.raises(ValueError):
        FingerprintConfig(min_observations_for_limited_history=10, min_observations_for_established_baseline=10)
    with pytest.raises(ValueError):
        FingerprintConfig(min_observations_for_limited_history=10, min_observations_for_established_baseline=5)


def test_to_dict_is_json_serializable() -> None:
    config = FingerprintConfig()
    d = config.to_dict()
    assert isinstance(d["events_path"], str)
    assert isinstance(d["facilities_path"], str)


def test_describe_rationale_covers_both_thresholds() -> None:
    rationale = FingerprintConfig().describe_rationale()
    assert "min_observations_for_limited_history" in rationale
    assert "min_observations_for_established_baseline" in rationale
    for value in rationale.values():
        assert len(value) > 20
