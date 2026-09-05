"""
Robust feature deviation for GIFT Stage I.4.

Produces a **robust deviation index**:

    abs(current - historical_median) / historical_MAD

when MAD > 0. This is deliberately NOT called a z-score and is NOT
assumed to follow a normal distribution. Stage I.3's MAD is the raw
median absolute deviation; this module never multiplies by 1.4826.

ZERO-MAD HANDLING (mandatory)
------------------------------------------------------------------------
Case 1: MAD == 0 and current == median → deviation = 0 (not anomalous).
Case 2: MAD == 0 and current != median:
  - If historical IQR > 0, use abs(current - median) / IQR as a
    documented quantile-range fallback (still a robust deviation index,
    not a z-score).
  - If historical values are truly constant (IQR also 0), assign
    ``config.zero_mad_constant_mismatch_deviation`` — a documented
    engineering constant meaning "differs from a constant historical
    baseline". No undocumented epsilon is used as a MAD substitute.

Missing current values or empty baselines return None (never 0.0).
Zero means "no deviation"; None means "cannot evaluate".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.anomaly_detection.config import AnomalyConfig
from src.fingerprinting.robust_stats import mad, median, quantile


@dataclass(frozen=True)
class BaselineStats:
    """Prior-only robust summary for one numeric feature."""

    count: int
    median: float | None
    mad: float | None
    p25: float | None
    p75: float | None
    p05: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None

    @property
    def iqr(self) -> float | None:
        if self.p25 is None or self.p75 is None:
            return None
        return float(self.p75 - self.p25)


@dataclass(frozen=True)
class FeatureDeviation:
    """Result of comparing one current value to a prior baseline."""

    deviation: float | None
    method: str | None  # "mad", "iqr_fallback", "constant_mismatch", or None
    baseline: BaselineStats | None
    is_above_median: bool | None = None
    outside_historical_range: bool | None = None


def compute_baseline_stats(values: Sequence[float] | np.ndarray) -> BaselineStats:
    """Build prior-only robust stats; empty input → count=0 and all None."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return BaselineStats(0, None, None, None, None, None, None, None, None)
    return BaselineStats(
        count=int(arr.size),
        median=median(arr),
        mad=mad(arr),
        p25=quantile(arr, 0.25),
        p75=quantile(arr, 0.75),
        p05=quantile(arr, 0.05),
        p95=quantile(arr, 0.95),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
    )


def robust_deviation(
    current: float | None,
    baseline: BaselineStats,
    config: AnomalyConfig,
) -> FeatureDeviation:
    """Compute a robust deviation index for one feature.

    Returns deviation=None when the current value is missing or the
    baseline has no valid observations.
    """
    if current is None or (isinstance(current, float) and np.isnan(current)):
        return FeatureDeviation(None, None, baseline if baseline.count else None)
    if baseline.count == 0 or baseline.median is None or baseline.mad is None:
        return FeatureDeviation(None, None, None)

    current_f = float(current)
    center = float(baseline.median)
    spread = float(baseline.mad)
    above = current_f > center
    outside = False
    if baseline.minimum is not None and baseline.maximum is not None:
        outside = current_f < baseline.minimum or current_f > baseline.maximum

    if spread > 0.0:
        deviation = abs(current_f - center) / spread
        return FeatureDeviation(float(deviation), "mad", baseline, above, outside)

    # MAD == 0
    if current_f == center or abs(current_f - center) <= 1e-15:
        return FeatureDeviation(0.0, "mad", baseline, False, False)

    iqr = baseline.iqr
    if iqr is not None and iqr > 0.0:
        deviation = abs(current_f - center) / iqr
        return FeatureDeviation(float(deviation), "iqr_fallback", baseline, above, outside)

    # Truly constant historical baseline; current differs.
    return FeatureDeviation(
        float(config.zero_mad_constant_mismatch_deviation),
        "constant_mismatch",
        baseline,
        above,
        True,
    )


def persistence_rarity_deviation(
    current_label: str | None,
    prior_labels: Sequence[str],
    *,
    min_prior: int,
) -> FeatureDeviation:
    """Categorical persistence deviation from prior label frequencies.

    Maps label rarity to a robust-deviation-like index:
      fraction >= 0.5  → 0.0 (common for this facility)
      fraction == 0    → 3.0 (never seen before among priors)
      otherwise        → linear interpolate between 0 and 3

    Returns None when fewer than ``min_prior`` prior labels exist or the
    current label is missing. Does NOT recompute Stage G.1 labels.
    """
    if current_label is None or (isinstance(current_label, float) and np.isnan(current_label)):
        return FeatureDeviation(None, None, None)
    labels = [str(x) for x in prior_labels if x is not None and str(x) != "" and str(x) != "nan"]
    if len(labels) < min_prior:
        return FeatureDeviation(None, None, None)

    current = str(current_label)
    match_count = sum(1 for lab in labels if lab == current)
    fraction = match_count / len(labels)
    if fraction >= 0.5:
        deviation = 0.0
    else:
        # fraction in [0, 0.5) → deviation in (0, 3]
        deviation = 3.0 * (1.0 - fraction / 0.5)
    return FeatureDeviation(float(deviation), "persistence_rarity", None, None, fraction == 0.0)
