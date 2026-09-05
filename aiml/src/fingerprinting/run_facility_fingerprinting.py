"""
Command-line entry point for GIFT Stage I.3 (Facility Fingerprinting &
Historical Thermal Baseline).

Usage (run from the ``aiml/`` directory, after Stage I.2 has produced
`thermal_events_with_facility_association.csv`):

    python -m src.fingerprinting.run_facility_fingerprinting

With explicit parameters:

    python -m src.fingerprinting.run_facility_fingerprinting \\
        --events data/processed/thermal_events_with_facility_association.csv \\
        --facilities data/processed/osm_facilities.csv \\
        --fingerprints-output data/processed/facility_thermal_fingerprints.csv \\
        --monthly-profile-output data/processed/facility_monthly_thermal_profile.csv \\
        --report data/processed/facility_fingerprinting_report.json \\
        --min-observations-for-limited-history 3 \\
        --min-observations-for-established-baseline 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.fingerprinting.fingerprint_config import FingerprintConfig
from src.fingerprinting.fingerprint_pipeline import load_events, load_facilities, run_facility_fingerprinting, save_outputs
from src.fingerprinting.fingerprint_report import save_report

_DEFAULT_EVENTS = Path("data/processed/thermal_events_with_facility_association.csv")
_DEFAULT_FACILITIES = Path("data/processed/osm_facilities.csv")
_DEFAULT_FINGERPRINTS_OUTPUT = Path("data/processed/facility_thermal_fingerprints.csv")
_DEFAULT_MONTHLY_PROFILE_OUTPUT = Path("data/processed/facility_monthly_thermal_profile.csv")
_DEFAULT_REPORT = Path("data/processed/facility_fingerprinting_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage I.3: descriptive historical thermal fingerprint per "
            "facility, built from Stage I.2's confirmed facility associations. "
            "Does not perform anomaly detection or source classification."
        )
    )
    parser.add_argument("--events", type=Path, default=_DEFAULT_EVENTS, help=f"Stage I.2 event/association table (default: {_DEFAULT_EVENTS}).")
    parser.add_argument("--facilities", type=Path, default=_DEFAULT_FACILITIES, help=f"Stage I.1 facility table (default: {_DEFAULT_FACILITIES}).")
    parser.add_argument("--fingerprints-output", type=Path, default=_DEFAULT_FINGERPRINTS_OUTPUT, help=f"Output CSV (default: {_DEFAULT_FINGERPRINTS_OUTPUT}).")
    parser.add_argument("--monthly-profile-output", type=Path, default=_DEFAULT_MONTHLY_PROFILE_OUTPUT, help=f"Output CSV (default: {_DEFAULT_MONTHLY_PROFILE_OUTPUT}).")
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT, help=f"Output report JSON (default: {_DEFAULT_REPORT}).")
    parser.add_argument("--min-observations-for-limited-history", type=int, default=FingerprintConfig.min_observations_for_limited_history)
    parser.add_argument("--min-observations-for-established-baseline", type=int, default=FingerprintConfig.min_observations_for_established_baseline)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    config = FingerprintConfig(
        min_observations_for_limited_history=args.min_observations_for_limited_history,
        min_observations_for_established_baseline=args.min_observations_for_established_baseline,
        events_path=args.events,
        facilities_path=args.facilities,
    )

    print(f"Loading events from {args.events} ...")
    events_df = load_events(args.events)
    print(f"Loaded {len(events_df):,} events.")
    print(f"Loading facilities from {args.facilities} ...")
    facilities_df = load_facilities(args.facilities)
    print(f"Loaded {len(facilities_df):,} facilities.")
    print(f"Fingerprint config: {config.to_dict()}")

    result = run_facility_fingerprinting(
        events_df, facilities_df, config, events_input_path=str(args.events), facilities_input_path=str(args.facilities)
    )

    save_outputs(result, args.fingerprints_output, args.monthly_profile_output)
    save_report(result.report, args.report)

    print()
    print(f"Wrote {len(result.fingerprints_df):,} facility fingerprints -> {args.fingerprints_output}")
    print(f"Wrote {len(result.monthly_profile_df):,} monthly-profile rows -> {args.monthly_profile_output}")
    print(f"Wrote report -> {args.report}")
    print()
    print(json.dumps(result.report["fingerprint_coverage"], indent=2))
    print(json.dumps(result.report["observation_statistics"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
