"""Unit tests for deterministic FIRMS observation hashing (no database)."""

from __future__ import annotations

import math

import pandas as pd

from app.services.observation_identity import (
    OBSERVATION_HASH_FIELDS,
    canonicalize_hash_value,
    compute_observation_hash,
)


def _sample(**overrides):
    row = {
        "latitude": "18.97",
        "longitude": "83.80",
        "bright_ti4": "330.1",
        "scan": "0.39",
        "track": "0.36",
        "acq_date": "2026-09-05",
        "acq_time": "0655",
        "satellite": "N20",
        "instrument": "VIIRS",
        "confidence": "n",
        "version": "2.0NRT",
        "bright_ti5": "296.5",
        "frp": "2.41",
        "daynight": "D",
        "type": "0",
    }
    row.update(overrides)
    return row


def test_hash_fields_are_explicit() -> None:
    assert "latitude" in OBSERVATION_HASH_FIELDS
    assert "acq_time" in OBSERVATION_HASH_FIELDS
    assert len(OBSERVATION_HASH_FIELDS) == 15


def test_same_observation_same_hash() -> None:
    a = compute_observation_hash(_sample())
    b = compute_observation_hash(_sample())
    assert a == b
    assert len(a) == 64


def test_same_location_different_time_different_hash() -> None:
    a = compute_observation_hash(_sample(acq_time="0655"))
    b = compute_observation_hash(_sample(acq_time="0710"))
    assert a != b


def test_different_values_different_hash() -> None:
    a = compute_observation_hash(_sample(frp="2.41"))
    b = compute_observation_hash(_sample(frp="9.99"))
    assert a != b


def test_null_handling_is_deterministic() -> None:
    assert canonicalize_hash_value(None) == ""
    assert canonicalize_hash_value(float("nan")) == ""
    assert canonicalize_hash_value(pd.NA) == ""
    assert canonicalize_hash_value("  ") == ""
    assert canonicalize_hash_value("nan") == ""

    h1 = compute_observation_hash(_sample(type=None))
    h2 = compute_observation_hash(_sample(type=""))
    h3 = compute_observation_hash(_sample(type=pd.NA))
    assert h1 == h2 == h3


def test_whitespace_canonicalization() -> None:
    a = compute_observation_hash(_sample(satellite="N20"))
    b = compute_observation_hash(_sample(satellite="  N20  "))
    assert a == b
