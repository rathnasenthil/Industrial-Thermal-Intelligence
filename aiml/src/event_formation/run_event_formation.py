"""
Command-line entry point for GIFT Stage G (Geospatial Event Formation).

Usage (run from the ``aiml/`` directory):

    python -m src.event_formation.run_event_formation

With explicit parameters:

    python -m src.event_formation.run_event_formation \\
        --input data/processed/firms_viirs_india_2023_2024_clean.csv \\
        --spatial-eps-km 1.5 --temporal-eps-hours 36 --min-samples 2 \\
        --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.event_formation.config import STDBSCANConfig
from src.event_formation.event_pipeline import (
    load_clean_detections,
    run_event_formation,
    save_outputs,
)
from src.event_formation.event_report import save_report

_DEFAULT_INPUT = Path("data/processed/firms_viirs_india_2023_2024_clean.csv")
_DEFAULT_OUTPUT_DIR = Path("data/processed")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GIFT Stage G: ST-DBSCAN geospatial-temporal thermal-event formation."
    )
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help=f"Cleaned FIRMS detections CSV (default: {_DEFAULT_INPUT}).")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help=f"Directory for output files (default: {_DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--spatial-eps-km", type=float, default=STDBSCANConfig.spatial_eps_km)
    parser.add_argument("--temporal-eps-hours", type=float, default=STDBSCANConfig.temporal_eps_hours)
    parser.add_argument("--min-samples", type=int, default=STDBSCANConfig.min_samples)
    parser.add_argument("--query-batch-size", type=int, default=STDBSCANConfig.query_batch_size)
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally limit to the first N rows (for quick runs/benchmarking).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    config = STDBSCANConfig(
        spatial_eps_km=args.spatial_eps_km,
        temporal_eps_hours=args.temporal_eps_hours,
        min_samples=args.min_samples,
        query_batch_size=args.query_batch_size,
    )

    print(f"Loading detections from {args.input} ...")
    detections = load_clean_detections(args.input, max_rows=args.max_rows)
    print(f"Loaded {len(detections):,} detections.")
    print(f"Clustering config: {config.to_dict()}")

    result = run_event_formation(detections, config, input_path=str(args.input))

    events_path = args.output_dir / "thermal_events.csv"
    detections_path = args.output_dir / "thermal_event_detections.csv"
    noise_path = args.output_dir / "thermal_event_noise.csv"
    report_path = args.output_dir / "event_formation_report.json"

    save_outputs(result, events_path, detections_path, noise_path)
    save_report(result.report, report_path)

    print()
    print(f"Wrote {len(result.events_df):,} events -> {events_path}")
    print(f"Wrote {len(result.detections_df):,} clustered detections -> {detections_path}")
    print(f"Wrote {len(result.noise_df):,} noise detections -> {noise_path}")
    print(f"Wrote report -> {report_path}")
    print()
    print(json.dumps(result.report["counts"], indent=2))
    print(json.dumps(result.report["performance"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
