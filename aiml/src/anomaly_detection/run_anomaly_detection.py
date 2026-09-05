"""
Command-line entry point for GIFT Stage I.4 (Temporal Deviation & Anomaly Detection).

Usage (from the ``aiml/`` directory, after Stage I.2 / I.3):

    python -m src.anomaly_detection.run_anomaly_detection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.anomaly_detection.anomaly_pipeline import (
    load_events,
    load_fingerprints,
    run_anomaly_detection,
    save_outputs,
)
from src.anomaly_detection.anomaly_report import save_report
from src.anomaly_detection.config import AnomalyConfig, DEFAULT_FEATURE_WEIGHTS

_DEFAULT_EVENTS = Path("data/processed/thermal_events_with_facility_association.csv")
_DEFAULT_FINGERPRINTS = Path("data/processed/facility_thermal_fingerprints.csv")
_DEFAULT_OUTPUT = Path("data/processed/thermal_events_with_anomaly_detection.csv")
_DEFAULT_REPORT = Path("data/processed/anomaly_detection_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage I.4: walk-forward temporal deviation / anomaly detection "
            "relative to each facility's prior confirmed associations. "
            "Does not classify sources as industrial fires."
        )
    )
    parser.add_argument("--events", type=Path, default=_DEFAULT_EVENTS)
    parser.add_argument("--fingerprints", type=Path, default=_DEFAULT_FINGERPRINTS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--min-observations-for-limited-history", type=int, default=3)
    parser.add_argument("--min-observations-for-established-baseline", type=int, default=10)
    parser.add_argument("--normal-max-score", type=float, default=2.0)
    parser.add_argument("--elevated-max-score", type=float, default=3.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = AnomalyConfig(
        min_observations_for_limited_history=args.min_observations_for_limited_history,
        min_observations_for_established_baseline=args.min_observations_for_established_baseline,
        normal_max_score=args.normal_max_score,
        elevated_max_score=args.elevated_max_score,
        feature_weights=dict(DEFAULT_FEATURE_WEIGHTS),
        events_path=args.events,
        fingerprints_path=args.fingerprints,
    )

    print(f"Loading events from {args.events} ...")
    events_df = load_events(args.events)
    print(f"Loaded {len(events_df):,} events.")
    print(f"Loading I.3 fingerprints from {args.fingerprints} (metadata only) ...")
    fingerprints_df = load_fingerprints(args.fingerprints)
    print(f"Loaded {len(fingerprints_df):,} facility fingerprints.")
    print(f"Anomaly config: {config.to_dict()}")

    result = run_anomaly_detection(
        events_df,
        config,
        fingerprints_df=fingerprints_df,
        events_input_path=str(args.events),
        fingerprints_input_path=str(args.fingerprints),
    )
    save_outputs(result, args.output)
    save_report(result.report, args.report)

    print()
    print(f"Wrote {len(result.events_df):,} events -> {args.output}")
    print(f"Wrote report -> {args.report}")
    print()
    print(json.dumps(result.report["anomaly_status_counts"], indent=2))
    print(json.dumps(result.report["anomaly_confidence_counts"], indent=2))
    print(json.dumps(result.report["performance"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
