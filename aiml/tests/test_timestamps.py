"""Tests for src.preprocessing.timestamps (date-format detection, HHMM parsing, UTC datetime)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.timestamps import build_acq_datetime, detect_acq_date_format


def test_detect_acq_date_format_iso() -> None:
    series = pd.Series(["2023-01-01", "2023-06-15", "2023-12-31"])
    assert detect_acq_date_format(series) == "%Y-%m-%d"


def test_detect_acq_date_format_day_first() -> None:
    """The task's originally-assumed DD-MM-YYYY format must still be supported."""
    series = pd.Series(["01-01-2023", "15-06-2023", "31-12-2023"])
    assert detect_acq_date_format(series) == "%d-%m-%Y"


def test_detect_acq_date_format_raises_on_unrecognizable_data() -> None:
    series = pd.Series(["not-a-date", "also-not-a-date"])
    with pytest.raises(ValueError):
        detect_acq_date_format(series)


@pytest.mark.parametrize(
    "acq_time,expected_hour,expected_minute",
    [
        (655, 6, 55),
        (712, 7, 12),
        (0, 0, 0),
        (5, 0, 5),
        (55, 0, 55),
        (2359, 23, 59),
    ],
)
def test_build_acq_datetime_parses_hhmm_correctly(acq_time, expected_hour, expected_minute) -> None:
    df = pd.DataFrame({"acq_date": ["2023-01-01"], "acq_time": [acq_time]})

    result = build_acq_datetime(df)

    assert result.valid_mask.iloc[0]
    ts = result.df["acq_datetime"].iloc[0]
    assert ts.hour == expected_hour
    assert ts.minute == expected_minute
    assert str(ts.tzinfo) == "UTC"


def test_build_acq_datetime_midnight_handling() -> None:
    df = pd.DataFrame({"acq_date": ["2023-01-01"], "acq_time": [0]})

    result = build_acq_datetime(df)

    ts = result.df["acq_datetime"].iloc[0]
    assert (ts.hour, ts.minute) == (0, 0)
    assert result.df["hour_utc"].iloc[0] == 0
    assert result.df["day_of_year"].iloc[0] == 1
    assert result.df["month"].iloc[0] == 1


def test_build_acq_datetime_derived_fields() -> None:
    # 2023-03-15 is a Wednesday (day_of_week == 2 under pandas' Monday=0 convention).
    df = pd.DataFrame({"acq_date": ["2023-03-15"], "acq_time": [1430]})

    result = build_acq_datetime(df)
    row = result.df.iloc[0]

    assert row["hour_utc"] == 14
    assert row["day_of_week"] == 2
    assert row["month"] == 3
    assert row["day_of_year"] == 74


@pytest.mark.parametrize("bad_time", [2400, 2500, 9999, -1])
def test_build_acq_datetime_flags_invalid_time_values(bad_time) -> None:
    df = pd.DataFrame({"acq_date": ["2023-01-01"], "acq_time": [bad_time]})

    result = build_acq_datetime(df)

    assert not result.valid_mask.iloc[0]
    assert pd.isna(result.df["acq_datetime"].iloc[0])
    assert result.stats["invalid_time_count"] == 1
    assert result.stats["invalid_date_count"] == 0


def test_build_acq_datetime_flags_invalid_date_values() -> None:
    # A large majority of well-formed YYYY-MM-DD dates so format detection
    # succeeds, plus one malformed date that must be flagged invalid rather
    # than silently dropped or guessed at.
    good_dates = [f"2023-{month:02d}-{day:02d}" for month in range(1, 13) for day in range(1, 21)]
    df = pd.DataFrame(
        {"acq_date": good_dates + ["2023-13-40"], "acq_time": [655] * len(good_dates) + [655]}
    )

    result = build_acq_datetime(df)

    assert result.valid_mask.iloc[:-1].all()
    assert not result.valid_mask.iloc[-1]
    assert result.stats["invalid_date_count"] == 1


def test_build_acq_datetime_does_not_convert_to_local_time() -> None:
    """The resulting datetime must stay in UTC, never shifted to IST."""
    df = pd.DataFrame({"acq_date": ["2023-01-01"], "acq_time": [655]})

    result = build_acq_datetime(df)
    ts = result.df["acq_datetime"].iloc[0]

    assert ts.utcoffset().total_seconds() == 0
    assert ts.hour == 6 and ts.minute == 55
