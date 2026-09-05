"""
Configuration for GIFT Stage I.3 (Facility Fingerprinting & Historical
Thermal Baseline).

The thresholds below decide how much confirmed historical evidence a
facility needs before its statistics should be read as an established
baseline. They are ENGINEERING defaults chosen for transparency and
conservatism, not scientifically validated minimum sample sizes for any
particular downstream statistical test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

NO_OBSERVATIONS = "NO_OBSERVATIONS"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
LIMITED_HISTORY = "LIMITED_HISTORY"
ESTABLISHED_BASELINE = "ESTABLISHED_BASELINE"

FINGERPRINT_STATUSES: tuple[str, ...] = (
    NO_OBSERVATIONS,
    INSUFFICIENT_HISTORY,
    LIMITED_HISTORY,
    ESTABLISHED_BASELINE,
)


@dataclass(frozen=True)
class FingerprintConfig:
    """Tunable thresholds for GIFT Stage I.3.

    Attributes:
        min_observations_for_limited_history: A facility with fewer
            confirmed associated events than this (but at least 1) is
            `INSUFFICIENT_HISTORY` rather than `LIMITED_HISTORY`.

            Rationale for the default (3): with 1-2 historical events
            there is essentially no way to tell "a coincidental one-off
            spatial match" apart from "the start of a real recurring
            pattern at this facility" -- deliberately conservative, and
            not a claim that 3 is a statistically sufficient sample size
            for any particular downstream inference.
        min_observations_for_established_baseline: A facility with at
            least this many confirmed associated events is
            `ESTABLISHED_BASELINE`; fewer (but >= `min_observations_for_
            limited_history`) is `LIMITED_HISTORY`.

            Rationale for the default (10): a round, transparent
            engineering threshold, not a scientifically validated
            minimum sample size for median/MAD/quantile stability. It is
            fully configurable and expected to be revisited once labeled
            outcomes are available to calibrate against.
        events_path: Default input -- the Stage I.2 output (all Stage
            G/G.1 event columns plus facility association columns).
        facilities_path: Default input -- the Stage I.1 normalized
            facility layer (the full facility universe, including
            facilities with zero associated events).
    """

    min_observations_for_limited_history: int = 3
    min_observations_for_established_baseline: int = 10
    events_path: Path = Path("data/processed/thermal_events_with_facility_association.csv")
    facilities_path: Path = Path("data/processed/osm_facilities.csv")

    def __post_init__(self) -> None:
        if self.min_observations_for_limited_history < 1:
            raise ValueError("min_observations_for_limited_history must be >= 1.")
        if self.min_observations_for_established_baseline <= self.min_observations_for_limited_history:
            raise ValueError(
                "min_observations_for_established_baseline must be strictly greater than "
                "min_observations_for_limited_history."
            )

    def classify_status(self, event_count: int) -> str:
        """Deterministic `fingerprint_status` for a facility's confirmed event count."""
        if event_count <= 0:
            return NO_OBSERVATIONS
        if event_count < self.min_observations_for_limited_history:
            return INSUFFICIENT_HISTORY
        if event_count < self.min_observations_for_established_baseline:
            return LIMITED_HISTORY
        return ESTABLISHED_BASELINE

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain, JSON-serializable dict."""
        d = asdict(self)
        d["events_path"] = str(d["events_path"])
        d["facilities_path"] = str(d["facilities_path"])
        return d

    def describe_rationale(self) -> dict[str, str]:
        """Human-readable rationale strings for each threshold (for reports)."""
        return {
            "min_observations_for_limited_history": (
                "Below this many confirmed associated events, a coincidental "
                "one-off spatial match cannot be distinguished from the start "
                "of a genuine recurring pattern; deliberately conservative, "
                "not a statistically validated minimum sample size."
            ),
            "min_observations_for_established_baseline": (
                "Round, transparent engineering threshold for calling a "
                "facility's historical statistics an 'established baseline'; "
                "not a scientifically validated minimum for median/MAD/"
                "quantile stability, and fully configurable."
            ),
        }


DEFAULT_CONFIG = FingerprintConfig()
