"""
Command-line entry point for GIFT Stage I.5 (NASA STA Evidence Integration).

Usage (from ``aiml/``):

    python -m src.sta_evidence.run_sta_integration

Requires locally supplied NASA STA Mask and/or Detections files under
``data/raw/`` (see README). Does not download or fabricate STA data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.sta_evidence.config import STAConfig
from src.sta_evidence.sta_loader import STASourceMissingError, resolve_existing_paths
from src.sta_evidence.sta_pipeline import load_events, run_sta_integration, save_outputs
from src.sta_evidence.sta_report import build_missing_source_report, save_report

_DEFAULT_EVENTS = Path("data/processed/thermal_events_with_anomaly_detection.csv")
_DEFAULT_EVENTS_OUT = Path("data/processed/thermal_events_with_sta_evidence.csv")
_DEFAULT_CANDIDATES = Path("data/processed/thermal_event_sta_candidates.csv")
_DEFAULT_STA_NORM = Path("data/processed/sta_normalized.csv")
_DEFAULT_REPORT = Path("data/processed/sta_integration_report.json")
_DEFAULT_MASK = Path("data/raw/nasa_firms_sta_mask.geojson")
_DEFAULT_DETECTION = Path("data/raw/nasa_firms_sta_detections.geojson")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage I.5: integrate NASA FIRMS Static Thermal Anomaly evidence "
            "with thermal events. STA is supporting evidence only — not industrial-fire classification."
        )
    )
    parser.add_argument("--events", type=Path, default=_DEFAULT_EVENTS)
    parser.add_argument("--mask", type=Path, default=_DEFAULT_MASK, help="Local STA Mask vector file.")
    parser.add_argument("--detections", type=Path, default=_DEFAULT_DETECTION, help="Local STA Detections vector file.")
    parser.add_argument("--events-output", type=Path, default=_DEFAULT_EVENTS_OUT)
    parser.add_argument("--candidates-output", type=Path, default=_DEFAULT_CANDIDATES)
    parser.add_argument("--sta-normalized-output", type=Path, default=_DEFAULT_STA_NORM)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--association-radius-km", type=float, default=1.0)
    parser.add_argument("--sta-source-version", type=str, default=None)
    parser.add_argument("--sta-download-date", type=str, default=None)
    parser.add_argument(
        "--allow-missing-source",
        action="store_true",
        help="Write a missing-source report and exit 0 instead of failing when no STA file exists.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = STAConfig(
        association_radius_km=args.association_radius_km,
        mask_path=args.mask,
        detection_path=args.detections,
        events_path=args.events,
        sta_source_version=args.sta_source_version,
        sta_download_date=args.sta_download_date,
    )

    print(f"STA config: {config.to_dict()}")
    existing = resolve_existing_paths(config)
    if not existing:
        print("ERROR: No local NASA STA source file found.")
        print(f"  Expected MASK: {config.mask_path}")
        print(f"  Expected DETECTIONS: {config.detection_path}")
        print(f"  Docs: {config.sta_documentation_url}")
        events_count = None
        if Path(args.events).exists():
            # Cheap row count without full parse of huge CSV if possible
            events_count = sum(1 for _ in open(args.events, "r", encoding="utf-8")) - 1
        report = build_missing_source_report(config, events_count=events_count)
        save_report(report, args.report)
        print(f"Wrote missing-source report -> {args.report}")
        if args.allow_missing_source:
            return 0
        return 2

    print(f"Loading events from {args.events} ...")
    events_df = load_events(args.events)
    print(f"Loaded {len(events_df):,} events.")
    print(f"STA sources: {[(str(p), layer) for p, layer in existing]}")

    try:
        result = run_sta_integration(events_df, config, events_input_path=str(args.events))
    except STASourceMissingError as exc:
        print(str(exc))
        save_report(build_missing_source_report(config, events_count=len(events_df)), args.report)
        return 2

    save_outputs(
        result,
        events_output_path=args.events_output,
        candidates_output_path=args.candidates_output,
        sta_normalized_output_path=args.sta_normalized_output,
    )
    save_report(result.report, args.report)

    print()
    print(f"Wrote events -> {args.events_output}")
    print(f"Wrote candidates -> {args.candidates_output}")
    print(f"Wrote normalized STA -> {args.sta_normalized_output}")
    print(f"Wrote report -> {args.report}")
    print(json.dumps(result.report["spatial_matching"], indent=2))
    print(json.dumps(result.report["evidence_quality_counts"], indent=2))
    print(json.dumps(result.report["performance"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
