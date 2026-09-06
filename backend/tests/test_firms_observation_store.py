"""Database tests for FIRMS observation store / deduplication."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import func, select, text

from app.models.firms_observation import FirmsObservation
from app.services.firms_observation_store import (
    FirmsObservationStoreError,
    observation_row_from_mapping,
    store_firms_observations,
    validate_coordinates,
)
from app.services.observation_identity import compute_observation_hash
from tests.conftest import REQUIRES_POSTGIS


def _obs(**overrides):
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
        "source_file": "firms_nrt_test",
    }
    row.update(overrides)
    return row


def test_invalid_coordinates_rejected_without_db() -> None:
    with pytest.raises(FirmsObservationStoreError, match="latitude"):
        validate_coordinates(100.0, 83.0)
    with pytest.raises(FirmsObservationStoreError, match="longitude"):
        validate_coordinates(18.0, 200.0)
    with pytest.raises(FirmsObservationStoreError):
        observation_row_from_mapping(_obs(latitude="999"))


@REQUIRES_POSTGIS
def test_duplicate_insertion_results_in_one_row(db_session) -> None:
    # Ensure table exists (migration applied).
    db_session.execute(text("SELECT 1 FROM firms_observations LIMIT 1"))

    marker = "phase2_dedup_test_unique_sat"
    row = _obs(satellite=marker, acq_time="1111")
    # Clean any prior test debris for this marker.
    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()

    first = store_firms_observations(db_session, [row, row], commit=True)
    assert first.received == 2
    assert first.inserted == 1
    assert first.duplicates == 1
    assert first.rejected == 0

    second = store_firms_observations(db_session, [row], commit=True)
    assert second.inserted == 0
    assert second.duplicates == 1

    count = db_session.scalar(
        select(func.count()).where(FirmsObservation.satellite == marker)
    )
    assert count == 1

    stored = db_session.scalar(
        select(FirmsObservation).where(FirmsObservation.satellite == marker)
    )
    assert stored is not None
    assert stored.event_id is None
    assert stored.observation_hash == compute_observation_hash(row)
    srid = db_session.execute(
        text(
            "SELECT ST_SRID(geometry) FROM firms_observations "
            "WHERE satellite = :s LIMIT 1"
        ),
        {"s": marker},
    ).scalar()
    assert int(srid) == 4326

    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()


@REQUIRES_POSTGIS
def test_multiple_new_observations_inserted(db_session) -> None:
    marker = "phase2_multi_insert_test"
    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()

    rows = [
        _obs(satellite=marker, acq_time="1200", frp="1.0"),
        _obs(satellite=marker, acq_time="1201", frp="2.0"),
        _obs(satellite=marker, acq_time="1202", frp="3.0"),
    ]
    result = store_firms_observations(db_session, rows, commit=True)
    assert result.inserted == 3
    assert result.duplicates == 0
    assert result.rejected == 0

    # event_id remains NULL for all
    null_events = db_session.scalar(
        select(func.count()).where(
            FirmsObservation.satellite == marker,
            FirmsObservation.event_id.is_(None),
        )
    )
    assert null_events == 3

    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()


@REQUIRES_POSTGIS
def test_invalid_coordinates_counted_as_rejected(db_session) -> None:
    marker = "phase2_reject_coords"
    good = _obs(satellite=marker, acq_time="1300")
    bad = _obs(satellite=marker, acq_time="1301", latitude="999")
    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()

    result = store_firms_observations(db_session, [good, bad], commit=True)
    assert result.received == 2
    assert result.inserted == 1
    assert result.rejected == 1

    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()


@REQUIRES_POSTGIS
def test_dataframe_input_supported(db_session) -> None:
    marker = "phase2_df_input"
    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()
    df = pd.DataFrame([_obs(satellite=marker, acq_time="1400")])
    result = store_firms_observations(db_session, df, commit=True)
    assert result.inserted == 1
    db_session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": marker},
    )
    db_session.commit()
