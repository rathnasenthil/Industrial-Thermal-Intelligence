"""
Exact-duplicate detection for FIRMS thermal detections.

Important: this module intentionally does NOT deduplicate on
latitude/longitude alone. Large or persistent thermal sources routinely
produce multiple *legitimate*, distinct records that share the same (or
very close) coordinates:

* A single overpass of a large fire can trigger several adjacent VIIRS
  pixels, each reported as its own detection with slightly different
  ``bright_ti4``/``frp``/``scan``/``track`` values.
* The same location can be re-detected on many different satellite
  overpasses (different ``acq_date``/``acq_time``), which is exactly the
  "persistent thermal source" signal that later Facility Fingerprinting
  depends on.
* Reported lat/lon in FIRMS is the pixel center, which can coincide across
  different overpasses without the underlying observation being a
  duplicate.

Treating same-location rows as duplicates would silently destroy the
persistence signal this project needs. Instead, a record is only
considered a duplicate if it is an **exact** match across every original
FIRMS column (i.e. it is almost certainly the same row appearing twice in
the source export, not two distinct satellite observations).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import pandas as pd

DEFAULT_DEDUPLICATION_STRATEGY = (
    "Rows are only treated as duplicates when they are identical across every "
    "original FIRMS column (latitude, longitude, bright_ti4, scan, track, "
    "acq_date, acq_time, satellite, instrument, confidence, version, "
    "bright_ti5, frp, daynight, type). Matching on latitude/longitude alone "
    "was deliberately avoided because large fires produce multiple adjacent "
    "VIIRS pixels and persistent sources are legitimately re-detected across "
    "many satellite overpasses at the same location — both are real, "
    "distinct observations that later stages (ST-DBSCAN event formation, "
    "Facility Fingerprinting) depend on, not noise to be collapsed."
)


class DuplicateDetectionResult(NamedTuple):
    """Result of exact-duplicate detection.

    Attributes:
        duplicate_mask: ``True`` for rows that are exact duplicates of an
            earlier row (i.e. every row after the first occurrence of a
            repeated exact combination).
        stats: ``{"exact_duplicate_count": int}`` — the number of rows
            that would be dropped if duplicates were removed, keeping only
            the first occurrence of each.
        strategy_note: Human-readable explanation of the deduplication
            strategy and why lat/lon-only matching was avoided (included
            verbatim in the preprocessing report).
    """

    duplicate_mask: pd.Series
    stats: dict[str, int]
    strategy_note: str


def detect_exact_duplicates(
    df: pd.DataFrame, subset: Sequence[str]
) -> DuplicateDetectionResult:
    """Flag exact duplicate rows based on the original FIRMS columns.

    Args:
        df: Combined FIRMS DataFrame.
        subset: Columns that must all match for two rows to be considered
            an exact duplicate. Should be the original FIRMS columns
            (not derived columns like ``acq_datetime``, which are a
            deterministic function of ``acq_date``/``acq_time`` anyway).

    Returns:
        A :class:`DuplicateDetectionResult`.
    """
    duplicate_mask = df.duplicated(subset=list(subset), keep="first")
    stats = {"exact_duplicate_count": int(duplicate_mask.sum())}
    return DuplicateDetectionResult(
        duplicate_mask=duplicate_mask,
        stats=stats,
        strategy_note=DEFAULT_DEDUPLICATION_STRATEGY,
    )
