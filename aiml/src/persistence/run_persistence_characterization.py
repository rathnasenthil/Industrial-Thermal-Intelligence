"""
Command-line entry point for GIFT Stage G.1 (Persistence & Recurrence
Characterization).

Usage (run from the ``aiml/`` directory, after Stage G has produced
``thermal_events.csv``):

    python -m src.persistence.run_persistence_characterization

With explicit parameters:

    python -m src.persistence.run_persistence_characterization \\
        --input data/processed/thermal_events.csv \\
        --output data/processed/thermal_events_with_persistence.csv \\
        --report data/processed/persistence_characterization_report.json \\
        --min-detections-for-classification 3 \\
        --short-lived-max-duration-hours 48 \\
        --persistent-min-duty-cycle 0.3 \\
        --persistent-max-gap-hours 72
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.persistence.config import PersistenceConfig
from src.persistence.persistence_pipeline import (
    load_thermal_events,
    run_persistence_characterization,
    save_events_with_persistence,
)
from src.persistence.persistence_report import save_report

_DEFAULT_INPUT = Path("data/processed/thermal_events.csv")
_DEFAULT_OUTPUT = Path("data/processed/thermal_events_with_persistence.csv")
_DEFAULT_REPORT = Path("data/processed/persistence_characterization_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GIFT Stage G.1: deterministic persistence/recurrence characterization of Stage G thermal events."
    )
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help=f"Stage G thermal_events.csv (default: {_DEFAULT_INPUT}).")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help=f"Output CSV path (default: {_DEFAULT_OUTPUT}).")
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT, help=f"Output report JSON path (default: {_DEFAULT_REPORT}).")
    parser.add_argument("--min-detections-for-classification", type=int, default=PersistenceConfig.min_detections_for_classification)
    parser.add_argument("--short-lived-max-duration-hours", type=float, default=PersistenceConfig.short_lived_max_duration_hours)
    parser.add_argument("--persistent-min-duty-cycle", type=float, default=PersistenceConfig.persistent_min_duty_cycle)
    parser.add_argument("--persistent-max-gap-hours", type=float, default=PersistenceConfig.persistent_max_gap_hours)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    config = PersistenceConfig(
        min_detections_for_classification=args.min_detections_for_classification,
        short_lived_max_duration_hours=args.short_lived_max_duration_hours,
        persistent_min_duty_cycle=args.persistent_min_duty_cycle,
        persistent_max_gap_hours=args.persistent_max_gap_hours,
    )

    print(f"Loading Stage G events from {args.input} ...")
    events_df = load_thermal_events(args.input)
    print(f"Loaded {len(events_df):,} events.")
    print(f"Classification config: {config.to_dict()}")

    result = run_persistence_characterization(events_df, config, input_path=str(args.input))

    save_events_with_persistence(result.events_df, args.output)
    save_report(result.report, args.report)

    print()
    print(f"Wrote {len(result.events_df):,} classified events -> {args.output}")
    print(f"Wrote report -> {args.report}")
    print()
    print(json.dumps(result.report["label_counts"], indent=2))
    print(json.dumps(result.report["label_percentages"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
