"""
CLI for GIFT Stage V — Independent Validation & Evaluation.

    python -m src.validation.run_validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.validation.config import ValidationConfig
from src.validation.validation_pipeline import load_events, run_validation, save_outputs

_DEFAULT_MATCHES = Path("data/processed/validation_event_matches.csv")
_DEFAULT_METRICS = Path("data/processed/validation_metrics.json")
_DEFAULT_REPORT = Path("data/processed/validation_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage V: independent validation of evidence-fusion candidates. "
            "Pipeline-derived evidence is never used as ground truth. "
            "Metrics are reported only when independent labels exist."
        )
    )
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--validation", type=Path, default=None, help="Independent labels CSV/GeoJSON.")
    parser.add_argument("--matches", type=Path, default=_DEFAULT_MATCHES)
    parser.add_argument("--metrics", type=Path, default=_DEFAULT_METRICS)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--spatial-tolerance-km", type=float, default=5.0)
    parser.add_argument("--temporal-tolerance-hours", type=float, default=72.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ValidationConfig(
        spatial_tolerance_km=args.spatial_tolerance_km,
        temporal_tolerance_hours=args.temporal_tolerance_hours,
    )
    events_path = Path(args.events) if args.events is not None else Path(config.events_path)
    if not events_path.exists():
        print(f"ERROR: events file not found: {events_path}")
        return 2

    print(f"Loading events from {events_path} ...")
    events_df = load_events(events_path)
    print(f"Loaded {len(events_df):,} events.")

    result = run_validation(
        events_df,
        config,
        validation_path=args.validation,
    )
    save_outputs(result, args.matches, args.metrics, args.report)

    print()
    print(f"Status: {result.report.get('status')}")
    print(f"Wrote matches -> {args.matches} ({len(result.matches_df):,} rows)")
    print(f"Wrote metrics -> {args.metrics}")
    print(f"Wrote report -> {args.report}")
    print(json.dumps({"metric_status": result.report.get("metric_status"), "processing_time_seconds": result.report.get("processing_time_seconds")}, indent=2))
    if result.report.get("warnings"):
        print("Warnings:")
        for w in result.report["warnings"]:
            try:
                print(f"  - {w}")
            except UnicodeEncodeError:
                print(f"  - {w.encode('ascii', 'replace').decode('ascii')}")
    if result.report.get("status") == "VALIDATION_DATA_UNAVAILABLE":
        print("NO VALIDATED PERFORMANCE CLAIM IS MADE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
