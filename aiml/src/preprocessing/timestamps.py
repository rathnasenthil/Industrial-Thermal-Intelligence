"""
Timestamp construction for FIRMS thermal detections.

Combines the FIRMS ``acq_date`` and ``acq_time`` columns into a single,
timezone-aware UTC ``acq_datetime`` column, plus a handful of derived
convenience fields (hour/day-of-week/month/day-of-year). Everything stays
in UTC at this stage — conversion to local (India) time is a job for a
later stage, not data-quality preprocessing.

Two things make this non-trivial and are handled explicitly here:

1. ``acq_date`` format is NOT assumed. The FIRMS archive has used both
   ``YYYY-MM-DD`` and ``DD-MM-YYYY`` in different export tools/eras, and
   the two are only distinguishable by inspecting the data (a leading
   4-digit token is unambiguous, e.g. "2023-01-01" is not a valid
   DD-MM-YYYY date since there is no month/day "2023"). This module
   auto-detects the format from the data rather than assuming one.
2. ``acq_time`` is a numeric HHMM-like value *without guaranteed zero
   padding* (e.g. ``0``, ``5``, ``55``, ``655``, ``712``, ``2359``). It is
   zero-padded to 4 digits and split into hour/minute rather than treated
   as seconds or as a plain integer offset.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

# Candidate acq_date formats seen across FIRMS archive export tools.
_CANDIDATE_DATE_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%d-%m-%Y")

# A detected format must successfully parse at least this fraction of a
# non-null sample to be accepted; otherwise we refuse to guess.
_MIN_FORMAT_MATCH_RATE = 0.99


class TimestampResult(NamedTuple):
    """Result of building the ``acq_datetime`` column and its derivatives.

    Attributes:
        df: Copy of the input DataFrame with ``acq_datetime`` (UTC,
            timezone-aware), ``hour_utc``, ``day_of_week`` (0=Monday,
            pandas ``dayofweek`` convention), ``month`` and ``day_of_year``
            columns added. Rows whose date/time could not be parsed have
            ``NaT``/``NA`` in all of these columns.
        valid_mask: ``True`` where a full, valid ``acq_datetime`` could be
            constructed.
        stats: Summary counts (see :func:`build_acq_datetime`).
        detected_date_format: The ``acq_date`` format that was detected
            and used (``"%Y-%m-%d"`` or ``"%d-%m-%Y"``).
    """

    df: pd.DataFrame
    valid_mask: pd.Series
    stats: dict[str, int]
    detected_date_format: str


def detect_acq_date_format(
    acq_date: pd.Series, sample_size: int = 5000
) -> str:
    """Detect whether ``acq_date`` uses ``YYYY-MM-DD`` or ``DD-MM-YYYY``.

    Rather than assuming a format, this samples the non-null values and
    checks which candidate format parses successfully for (almost) all of
    them.

    Args:
        acq_date: The raw ``acq_date`` column (string dtype).
        sample_size: Maximum number of non-null values to sample for
            format detection (the full column is then parsed with the
            winning format).

    Returns:
        The winning strftime format string.

    Raises:
        ValueError: If neither candidate format reliably parses the data,
            meaning the schema truly differs from what this module
            supports and must not be silently guessed.
    """
    non_null = acq_date.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        raise ValueError("acq_date column has no non-null values to detect a format from.")

    sample = non_null.sample(min(sample_size, len(non_null)), random_state=0)

    best_format = None
    best_rate = 0.0
    for fmt in _CANDIDATE_DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_rate = rate
            best_format = fmt

    if best_format is None or best_rate < _MIN_FORMAT_MATCH_RATE:
        raise ValueError(
            "Could not reliably detect acq_date format. Tried "
            f"{_CANDIDATE_DATE_FORMATS}; best match rate was {best_rate:.2%}. "
            "Refusing to guess - please verify the acq_date schema."
        )
    return best_format


def _parse_acq_time(acq_time: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Parse a numeric HHMM-like ``acq_time`` column into hour/minute.

    Values are treated as HHMM (not seconds, not zero-padded strings that
    must match a fixed width): ``0`` -> 00:00, ``5`` -> 00:05,
    ``55`` -> 00:55, ``655`` -> 06:55, ``712`` -> 07:12, ``2359`` -> 23:59.

    Args:
        acq_time: The raw ``acq_time`` column (string or numeric dtype).

    Returns:
        A tuple ``(hour, minute, valid_mask)`` where ``hour``/``minute``
        are nullable ``Int64`` Series and ``valid_mask`` is ``True`` for
        rows with a parsable, in-range (00:00-23:59) time.
    """
    numeric = pd.to_numeric(acq_time, errors="coerce")

    parsable_mask = numeric.notna()
    # HHMM must be an integer in [0, 2359].
    in_range_mask = parsable_mask & (numeric >= 0) & (numeric <= 2359)

    numeric_int = numeric.round().astype("Int64")
    hour = (numeric_int // 100).astype("Int64")
    minute = (numeric_int % 100).astype("Int64")

    minute_valid_mask = in_range_mask & (minute <= 59)
    valid_mask = in_range_mask & minute_valid_mask

    hour = hour.where(valid_mask)
    minute = minute.where(valid_mask)
    return hour, minute, valid_mask


def build_acq_datetime(
    df: pd.DataFrame, date_format: str | None = None
) -> TimestampResult:
    """Build the ``acq_datetime`` (UTC) column and derived time fields.

    Args:
        df: DataFrame containing raw ``acq_date`` (string, e.g.
            ``2023-01-01`` or ``01-01-2023``) and ``acq_time`` (numeric
            HHMM-like, e.g. ``655``) columns.
        date_format: Optional explicit strftime format for ``acq_date``.
            If ``None`` (default), the format is auto-detected via
            :func:`detect_acq_date_format`.

    Returns:
        A :class:`TimestampResult`. ``acq_datetime`` is timezone-aware UTC
        (never converted to local time at this stage). Rows with an
        unparsable date or an out-of-range time get ``NaT``/``NA`` in all
        derived columns and ``valid_mask=False``.
    """
    out = df.copy()

    detected_format = date_format or detect_acq_date_format(out["acq_date"])
    date_part = pd.to_datetime(
        out["acq_date"].astype(str).str.strip(), format=detected_format, errors="coerce"
    )
    date_valid_mask = date_part.notna()

    hour, minute, time_valid_mask = _parse_acq_time(out["acq_time"])

    valid_mask = date_valid_mask & time_valid_mask

    time_offset = pd.to_timedelta(hour.astype("float64"), unit="h") + pd.to_timedelta(
        minute.astype("float64"), unit="m"
    )
    acq_datetime = date_part + time_offset
    acq_datetime = acq_datetime.where(valid_mask)
    acq_datetime = acq_datetime.dt.tz_localize("UTC")

    out["acq_datetime"] = acq_datetime
    out["hour_utc"] = acq_datetime.dt.hour.astype("Int64")
    out["day_of_week"] = acq_datetime.dt.dayofweek.astype("Int64")
    out["month"] = acq_datetime.dt.month.astype("Int64")
    out["day_of_year"] = acq_datetime.dt.dayofyear.astype("Int64")

    stats = {
        "invalid_date_count": int((~date_valid_mask).sum()),
        "invalid_time_count": int((~time_valid_mask).sum()),
        "invalid_timestamp_count": int((~valid_mask).sum()),
    }
    return TimestampResult(
        df=out, valid_mask=valid_mask, stats=stats, detected_date_format=detected_format
    )
