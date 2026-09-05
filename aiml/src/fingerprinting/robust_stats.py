"""
Robust descriptive-statistics helpers for GIFT Stage I.3 (Facility
Fingerprinting & Historical Thermal Baseline).

WHY ROBUST STATISTICS, NOT THE MEAN
------------------------------------------------------------------------
FRP, event size and event duration distributions in this dataset are
heavily right-skewed (a small number of very large/persistent events
alongside many brief, small ones). A single extreme value can move an
arithmetic mean far more than it should for a *descriptive* baseline, so
this module favors:

* the median (middle value; insensitive to extreme outliers), and
* the MAD -- Median Absolute Deviation, ``median(|x - median(x)|)`` --
  as a robust spread measure, instead of the standard deviation.

IMPORTANT: this module's `mad()` is the RAW median absolute deviation.
It is deliberately NOT multiplied by the usual normal-consistency scale
factor (~1.4826) that would let it be read "as if" it were a standard
deviation -- doing that silently would misrepresent a robust, distribution
-free spread measure as a parametric one. Any caller wanting a normal-
consistent scaled MAD must do that conversion explicitly and label it as
such; this module never does it implicitly.

This module intentionally does NOT discard extreme values (e.g. a very
large peak FRP) -- `max()` is reported precisely to preserve that
historical information; only the *central tendency/spread* statistics
(median/MAD) are chosen to be outlier-resistant.

MISSING DATA
------------------------------------------------------------------------
Every function here drops NaN values before computing anything. An
all-NaN or empty input returns ``None`` for every statistic (never `0.0`
or a fabricated placeholder) -- "no valid observations" and "the
observed value happens to be zero" must never be confused (see the
Stage I.3 module docstring in `facility_fingerprint.py` for why this
matters for a historical baseline).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Percentiles calculated for every "*_p25/p75/p90" fingerprint field.
QUANTILES: dict[str, float] = {"p25": 0.25, "p75": 0.75, "p90": 0.90}


def _clean(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Convert to a 1-D float array with NaN/None/missing values dropped."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[~np.isnan(arr)]


def median(values: Sequence[float] | np.ndarray | pd.Series) -> float | None:
    """Median of the non-missing values, or ``None`` if none exist."""
    clean = _clean(values)
    if clean.size == 0:
        return None
    return float(np.median(clean))


def mad(values: Sequence[float] | np.ndarray | pd.Series) -> float | None:
    """Raw median absolute deviation from the median (see module docstring).

    A single valid observation has ``MAD == 0.0`` (it has no spread
    around itself) -- mathematically correct for descriptive purposes,
    even though `fingerprint_status` must separately flag such a
    facility as having insufficient history (this module has no
    knowledge of that threshold; see `fingerprint_config.py`).
    """
    clean = _clean(values)
    if clean.size == 0:
        return None
    center = np.median(clean)
    return float(np.median(np.abs(clean - center)))


def quantile(values: Sequence[float] | np.ndarray | pd.Series, q: float) -> float | None:
    """`q`-th quantile (0-1) of the non-missing values, or ``None`` if none exist."""
    clean = _clean(values)
    if clean.size == 0:
        return None
    return float(np.quantile(clean, q))


def summary_stats(values: Sequence[float] | np.ndarray | pd.Series, prefix: str) -> dict[str, float | None]:
    """Full `{prefix}_median/mad/p25/p75/p90/max` dict for one array of values.

    Every value is ``None`` (never `0.0`) if `values` has no non-missing
    entries at all.
    """
    clean = _clean(values)
    result: dict[str, float | None] = {
        f"{prefix}_median": median(clean),
        f"{prefix}_mad": mad(clean),
    }
    for suffix, q in QUANTILES.items():
        result[f"{prefix}_{suffix}"] = quantile(clean, q)
    result[f"{prefix}_max"] = float(np.max(clean)) if clean.size else None
    return result


def grouped_summary_stats(df: pd.DataFrame, group_col: str, value_col: str, prefix: str) -> pd.DataFrame:
    """Vectorized per-group `{prefix}_median/mad/p25/p75/p90/max`.

    Fully vectorized via `pandas.groupby` (median/quantile natively,
    MAD via a `transform` + a second `groupby`) -- there is deliberately
    no per-group Python loop or `.apply(median)`-style call, since Stage
    I.3 must scale to tens of thousands of facility groups (see module
    docstring in `facility_fingerprint.py` for the efficiency
    requirement inherited from the task spec).

    Args:
        df: Any DataFrame containing `group_col` and `value_col`.
        group_col: Column to group by (e.g. ``facility_id``).
        value_col: Numeric column to summarize (e.g. ``peak_frp``).
        prefix: Output column prefix (e.g. ``"peak_frp"``).

    Returns:
        A DataFrame indexed by every distinct value of `group_col` that
        appears in `df` (even if all of that group's `value_col` entries
        are NaN -- such a group gets all-`None`/NaN statistics, not a
        fabricated `0.0`), with columns
        ``{prefix}_median/mad/p25/p75/p90/max``.
    """
    numeric = pd.to_numeric(df[value_col], errors="coerce")
    grouped = numeric.groupby(df[group_col])

    group_median = grouped.transform("median")
    abs_dev = (numeric - group_median).abs()
    mad_by_group = abs_dev.groupby(df[group_col]).median()

    out = pd.DataFrame(
        {
            f"{prefix}_median": grouped.median(),
            f"{prefix}_mad": mad_by_group,
            f"{prefix}_p25": grouped.quantile(0.25),
            f"{prefix}_p75": grouped.quantile(0.75),
            f"{prefix}_p90": grouped.quantile(0.90),
            f"{prefix}_max": grouped.max(),
        }
    )
    # A group whose value_col is entirely NaN yields pandas NaN (not a
    # fabricated 0) from median/quantile/max above -- explicit for clarity.
    return out
