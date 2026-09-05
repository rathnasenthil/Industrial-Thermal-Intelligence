"""
End-to-end Facility Fingerprinting pipeline (GIFT Stage I.3).

Wires together, in order:

1. Load the Stage I.2 event/association table and the Stage I.1 facility
   table (both immutable baselines -- never rewritten, never
   re-derived: no spatial join is repeated here, and
   `thermal_event_detections.csv` is never read -- see
   `facility_fingerprint.py` module docstring).
2. Build one fingerprint row per facility (`facility_fingerprint.py`).
3. Build the normalized monthly activity profile (`monthly_profile.py`).
4. Assemble a JSON report (`fingerprint_report.py`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.fingerprinting.facility_fingerprint import REQUIRED_EVENT_COLUMNS, build_facility_fingerprints
from src.fingerprinting.fingerprint_config import FingerprintConfig
from src.fingerprinting.fingerprint_report import build_fingerprint_report
from src.fingerprinting.monthly_profile import build_monthly_profile

REQUIRED_FACILITY_COLUMNS: tuple[str, ...] = ("facility_id", "facility_name", "facility_type")


@dataclass
class FingerprintResult:
    """Result of running the Stage I.3 pipeline.

    Attributes:
        fingerprints_df: One row per input facility (see
            `facility_fingerprint.build_facility_fingerprints`).
        monthly_profile_df: Long-format (facility_id, month) activity table.
        report: JSON-serializable Stage I.3 report.
    """

    fingerprints_df: pd.DataFrame
    monthly_profile_df: pd.DataFrame
    report: dict[str, Any]


def load_events(path: str | Path) -> pd.DataFrame:
    """Load the Stage I.2 event/association table (read-only)."""
    events_path = Path(path)
    if not events_path.exists():
        raise FileNotFoundError(f"Thermal events (with facility association) file not found: {events_path}")
    df = pd.read_csv(events_path)
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{events_path}' is missing required column(s): {missing}. "
            "Stage I.3 expects the output of src.infrastructure.run_facility_association "
            "(thermal_events_with_facility_association.csv)."
        )
    return df


def load_facilities(path: str | Path) -> pd.DataFrame:
    """Load the Stage I.1 normalized facility table (read-only)."""
    facilities_path = Path(path)
    if not facilities_path.exists():
        raise FileNotFoundError(f"Stage I.1 facility file not found: {facilities_path}")
    df = pd.read_csv(facilities_path)
    missing = [c for c in REQUIRED_FACILITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"'{facilities_path}' is missing required column(s): {missing}.")
    return df


def run_facility_fingerprinting(
    events_df: pd.DataFrame,
    facilities_df: pd.DataFrame,
    config: FingerprintConfig,
    events_input_path: str = "<in-memory>",
    facilities_input_path: str = "<in-memory>",
) -> FingerprintResult:
    """Run the full Stage I.3 pipeline over already-loaded inputs.

    Args:
        events_df: Output of `load_events` (or an equivalent DataFrame).
        facilities_df: Output of `load_facilities` (or an equivalent DataFrame).
        config: `FingerprintConfig`.
        events_input_path: Recorded in the report for provenance only.
        facilities_input_path: Recorded in the report for provenance only.

    Returns:
        A `FingerprintResult`.
    """
    start_time = time.perf_counter()

    fingerprints_df = build_facility_fingerprints(events_df, facilities_df, config)
    monthly_profile_df = build_monthly_profile(events_df)

    # Invariant: every facility remains represented, exactly once.
    assert len(fingerprints_df) == facilities_df["facility_id"].nunique()
    assert fingerprints_df["facility_id"].is_unique
    assert set(fingerprints_df["facility_id"]) == set(facilities_df["facility_id"])

    processing_seconds = time.perf_counter() - start_time

    report = build_fingerprint_report(
        config=config,
        events_input_path=str(events_input_path),
        facilities_input_path=str(facilities_input_path),
        facility_count=facilities_df["facility_id"].nunique(),
        event_count=len(events_df),
        events_with_association_df=events_df,
        fingerprints_df=fingerprints_df,
        processing_seconds=processing_seconds,
    )

    return FingerprintResult(fingerprints_df=fingerprints_df, monthly_profile_df=monthly_profile_df, report=report)


def save_outputs(result: FingerprintResult, fingerprints_output_path: str | Path, monthly_profile_output_path: str | Path) -> None:
    """Write the Stage I.3 outputs to CSV."""
    fingerprints_path = Path(fingerprints_output_path)
    fingerprints_path.parent.mkdir(parents=True, exist_ok=True)
    result.fingerprints_df.to_csv(fingerprints_path, index=False)

    monthly_path = Path(monthly_profile_output_path)
    monthly_path.parent.mkdir(parents=True, exist_ok=True)
    result.monthly_profile_df.to_csv(monthly_path, index=False)
