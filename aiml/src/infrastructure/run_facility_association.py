"""
Command-line entry point for GIFT Stage I.2 (Thermal Event <-> Facility
Association).

Usage (run from the ``aiml/`` directory, after Stage G.1 and Stage I.1
have produced their outputs):

    python -m src.infrastructure.run_facility_association

With explicit parameters:

    python -m src.infrastructure.run_facility_association \\
        --events data/processed/thermal_events_with_persistence.csv \\
        --facilities data/processed/osm_facilities.geojson \\
        --events-output data/processed/thermal_events_with_facility_association.csv \\
        --candidates-output data/processed/thermal_event_facility_candidates.csv \\
        --report data/processed/facility_association_report.json \\
        --association-radius-km 5.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.infrastructure.association_config import AssociationConfig
from src.infrastructure.association_pipeline import load_events, run_facility_association, save_outputs
from src.infrastructure.association_report import save_report

_DEFAULT_EVENTS = Path("data/processed/thermal_events_with_persistence.csv")
_DEFAULT_FACILITIES = Path("data/processed/osm_facilities.geojson")
_DEFAULT_EVENTS_OUTPUT = Path("data/processed/thermal_events_with_facility_association.csv")
_DEFAULT_CANDIDATES_OUTPUT = Path("data/processed/thermal_event_facility_candidates.csv")
_DEFAULT_REPORT = Path("data/processed/facility_association_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage I.2: spatial-index-based association of thermal "
            "events with the Stage I.1 normalized OSM facility layer. "
            "This is a geospatial association step only -- it does not "
            "classify a thermal event's source."
        )
    )
    parser.add_argument("--events", type=Path, default=_DEFAULT_EVENTS, help=f"Event table (default: {_DEFAULT_EVENTS}).")
    parser.add_argument("--facilities", type=Path, default=_DEFAULT_FACILITIES, help=f"Facility table (default: {_DEFAULT_FACILITIES}).")
    parser.add_argument("--events-output", type=Path, default=_DEFAULT_EVENTS_OUTPUT, help=f"Output events CSV (default: {_DEFAULT_EVENTS_OUTPUT}).")
    parser.add_argument("--candidates-output", type=Path, default=_DEFAULT_CANDIDATES_OUTPUT, help=f"Output candidates CSV (default: {_DEFAULT_CANDIDATES_OUTPUT}).")
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT, help=f"Output report JSON (default: {_DEFAULT_REPORT}).")
    parser.add_argument("--association-radius-km", type=float, default=AssociationConfig.association_radius_km)
    parser.add_argument("--ambiguity-distance-tolerance-km", type=float, default=AssociationConfig.ambiguity_distance_tolerance_km)
    parser.add_argument("--max-candidates-per-event", type=int, default=AssociationConfig.max_candidates_per_event)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    config = AssociationConfig(
        association_radius_km=args.association_radius_km,
        ambiguity_distance_tolerance_km=args.ambiguity_distance_tolerance_km,
        max_candidates_per_event=args.max_candidates_per_event,
        events_path=args.events,
        facilities_path=args.facilities,
    )

    print(f"Loading events from {args.events} ...")
    events_df = load_events(args.events)
    print(f"Loaded {len(events_df):,} events.")
    print(f"Loading facilities from {args.facilities} ...")
    print(f"Association config: {config.to_dict()}")

    result = run_facility_association(events_df, args.facilities, config, events_input_path=str(args.events))

    save_outputs(result, args.events_output, args.candidates_output)
    save_report(result.report, args.report)

    print()
    print(f"Wrote {len(result.events_df):,} events with association -> {args.events_output}")
    print(f"Wrote {len(result.candidates_df):,} candidate rows -> {args.candidates_output}")
    print(f"Wrote report -> {args.report}")
    print()
    print(json.dumps(result.report["association_results"], indent=2))
    print(json.dumps(result.report["confidence_counts"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
