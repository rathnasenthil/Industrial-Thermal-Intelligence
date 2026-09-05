"""
Command-line entry point for GIFT Stage I.1 (OSM Facility Ingestion & Normalization).

Usage (run from the ``aiml/`` directory):

    python -m src.infrastructure.run_osm_ingestion

This auto-discovers a static OSM extract in ``data/raw/`` (see
`osm_loader.discover_default_osm_input`). If none is found, it does NOT
fail or fabricate data -- it still runs and writes a report that clearly
states no production OSM input was available.

With an explicit input file:

    python -m src.infrastructure.run_osm_ingestion --input data/raw/osm_facilities_india.geojson
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.infrastructure.config import InfrastructureConfig
from src.infrastructure.facility_report import save_report
from src.infrastructure.infrastructure_pipeline import run_infrastructure_ingestion, save_outputs
from src.infrastructure.osm_loader import discover_default_osm_input

_DEFAULT_RAW_DIR = Path("data/raw")
_DEFAULT_OUTPUT_DIR = Path("data/processed")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GIFT Stage I.1: OSM facility ingestion & normalization (static extract, no live Overpass dependency)."
    )
    parser.add_argument("--input", type=Path, default=None, help="Path to a static OSM extract (GeoJSON or CSV). If omitted, auto-discovers one in --raw-dir.")
    parser.add_argument("--raw-dir", type=Path, default=_DEFAULT_RAW_DIR, help=f"Directory to auto-discover a static extract in (default: {_DEFAULT_RAW_DIR}).")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help=f"Directory for output files (default: {_DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--source-version", type=str, default=None, help="Optional provenance string; defaults to '<filename> (mtime=...)'.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    config = InfrastructureConfig(raw_dir=args.raw_dir)

    input_path = args.input
    if input_path is None:
        input_path = discover_default_osm_input(args.raw_dir)

    if input_path is None:
        print(f"No static OSM extract found in '{args.raw_dir}' (and none passed via --input).")
        print("Proceeding with an EMPTY facility layer -- this is not an error.")
        print("See aiml/README.md (GIFT Stage I.1) for where to place a real extract.")
    else:
        print(f"Loading static OSM extract from {input_path} ...")

    result = run_infrastructure_ingestion(input_path, config, source_version=args.source_version)

    facilities_csv_path = args.output_dir / "osm_facilities.csv"
    facilities_geojson_path = args.output_dir / "osm_facilities.geojson"
    rejected_csv_path = args.output_dir / "osm_facilities_rejected.csv"
    report_path = args.output_dir / "osm_facility_report.json"

    save_outputs(result, facilities_csv_path, facilities_geojson_path, rejected_csv_path)
    save_report(result.report, report_path)

    print()
    print(f"Wrote {len(result.facilities_gdf):,} normalized facilities -> {facilities_csv_path}")
    print(f"Wrote {len(result.facilities_gdf):,} normalized facilities -> {facilities_geojson_path}")
    if not result.rejected_df.empty:
        print(f"Wrote {len(result.rejected_df):,} rejected records -> {rejected_csv_path}")
    print(f"Wrote report -> {report_path}")
    print()
    print(json.dumps(result.report["input"], indent=2))
    print(json.dumps(result.report["normalization"]["facility_type_counts"], indent=2))
    print(json.dumps(result.report["validation"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
