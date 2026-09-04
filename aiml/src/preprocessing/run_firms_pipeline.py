"""
Command-line entry point for the FIRMS ingestion + data-quality
preprocessing pipeline.

Usage (run from the ``aiml/`` directory so the ``src`` package resolves,
matching this project's existing test-suite convention):

    python -m src.preprocessing.run_firms_pipeline

With explicit input files:

    python -m src.preprocessing.run_firms_pipeline \\
        --input data/raw/viirs-jpss1_2023_India.csv \\
        --input data/raw/viirs-jpss1_2024_India.csv \\
        --output data/processed/firms_viirs_india_2023_2024_clean.csv \\
        --report data/processed/firms_preprocessing_report.json

If ``--input`` is omitted, the script looks in ``data/raw/`` for FIRMS CSV
files (matching either the documented naming convention
``firms_viirs_india_<year>.csv`` or the real archive-export naming
convention ``viirs-*_<year>_*.csv``) and uses whatever it finds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.preprocessing.firms_pipeline import run_firms_preprocessing, save_clean_dataset
from src.preprocessing.report import save_report

_DEFAULT_RAW_DIR = Path("data/raw")
_DEFAULT_OUTPUT_CSV = Path("data/processed/firms_viirs_india_2023_2024_clean.csv")
_DEFAULT_OUTPUT_REPORT = Path("data/processed/firms_preprocessing_report.json")

# Filename patterns to look for, in priority order, when --input is not
# given explicitly. The first pattern matches the naming convention
# documented for this project; the second matches the real NASA FIRMS
# archive-download naming convention actually found in data/raw/.
_DISCOVERY_PATTERNS: tuple[str, ...] = (
    "firms_viirs_india_*.csv",
    "viirs*_*_India.csv",
    "viirs*_*_india.csv",
)


def discover_default_inputs(raw_dir: Path) -> list[Path]:
    """Find FIRMS CSV files in ``raw_dir`` when no explicit input is given.

    Args:
        raw_dir: Directory to search (typically ``aiml/data/raw``).

    Returns:
        A sorted list of matching CSV paths (deduplicated across
        patterns).

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    found: dict[str, Path] = {}
    for pattern in _DISCOVERY_PATTERNS:
        for path in raw_dir.glob(pattern):
            found[path.name] = path
    if not found:
        raise FileNotFoundError(
            f"No FIRMS CSV files found in '{raw_dir}'. Expected files matching "
            f"one of {_DISCOVERY_PATTERNS}. Pass --input explicitly instead."
        )
    return sorted(found.values(), key=lambda p: p.name)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest and preprocess NASA FIRMS VIIRS CSV exports for the GIFT pipeline."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=None,
        help=(
            "Path to a raw FIRMS CSV file. Repeat for multiple files "
            "(e.g. --input a.csv --input b.csv). If omitted, files are "
            "auto-discovered under data/raw/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_CSV,
        help=f"Output path for the cleaned combined CSV (default: {_DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_DEFAULT_OUTPUT_REPORT,
        help=f"Output path for the JSON preprocessing report (default: {_DEFAULT_OUTPUT_REPORT}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_DEFAULT_RAW_DIR,
        help=f"Directory to auto-discover input files from when --input is omitted (default: {_DEFAULT_RAW_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    input_paths = [Path(p) for p in args.inputs] if args.inputs else discover_default_inputs(args.raw_dir)

    print(f"Input files ({len(input_paths)}):")
    for p in input_paths:
        print(f"  - {p}")

    result = run_firms_preprocessing(input_paths)
    save_clean_dataset(result.clean_df, args.output)
    save_report(result.report, args.report)

    print()
    print(f"Wrote cleaned dataset -> {args.output} ({len(result.clean_df)} rows)")
    print(f"Wrote preprocessing report -> {args.report}")
    print()
    print(json.dumps(result.report["row_counts"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
