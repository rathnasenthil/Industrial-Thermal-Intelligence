"""
Benchmark ST-DBSCAN event formation on representative subsets before
committing to a full ~1.17M-row run (per project requirement: "do not
silently run an extremely expensive algorithm").

Usage (from the ``aiml/`` directory):

    python -m src.event_formation.benchmark_st_dbscan

This measures processing time and peak memory (via `tracemalloc`) on:

* increasing chronological prefixes of the cleaned dataset (10k / 50k /
  200k rows), to observe how runtime scales with size, and
* a fixed-size sample drawn from a known high fire-density period
  (Oct-Nov, Indian crop-residue/stubble-burning season) restricted to a
  small bounding box, which stresses the "many detections close together
  in space and time" case that a naive spatial-only pre-filter could
  handle badly.

Results are written to ``data/processed/event_formation_benchmark.json``
and printed as a table. This script does NOT write thermal_events.csv/
thermal_event_detections.csv/thermal_event_noise.csv — use
`run_event_formation.py` for the real run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.event_formation.config import STDBSCANConfig
from src.event_formation.event_pipeline import load_clean_detections, run_event_formation

_DEFAULT_INPUT = Path("data/processed/firms_viirs_india_2023_2024_clean.csv")
_DEFAULT_OUTPUT = Path("data/processed/event_formation_benchmark.json")

# Punjab/Haryana bounding box: the best-known Indian crop-residue
# ("stubble") burning hotspot, active roughly October-November. Used to
# construct a *dense* benchmark scenario rather than a purely random
# subset, which would likely understate real-world worst-case density.
_DENSE_REGION_BBOX = {"lat_min": 29.0, "lat_max": 31.5, "lon_min": 74.0, "lon_max": 77.5}
_DENSE_MONTHS = (10, 11)


def _run_one(name: str, df: pd.DataFrame, config: STDBSCANConfig) -> dict[str, Any]:
    print(f"--- Benchmark: {name} ({len(df):,} rows) ---")
    if df.empty:
        print("  (no rows available for this scenario, skipping)")
        return {"scenario": name, "rows": 0, "skipped": True}

    result = run_event_formation(df, config, input_path=f"<benchmark:{name}>")
    perf = result.report["performance"]
    counts = result.report["counts"]
    print(
        f"  rows={len(df):,} time={perf['processing_seconds']}s "
        f"peak_mem={perf['peak_memory_mb']}MB events={counts['event_count']} "
        f"noise={counts['noise_detection_count']}"
    )
    return {
        "scenario": name,
        "rows": len(df),
        "processing_seconds": perf["processing_seconds"],
        "peak_memory_mb": perf["peak_memory_mb"],
        "event_count": counts["event_count"],
        "noise_detection_count": counts["noise_detection_count"],
        "percent_clustered": counts["percent_clustered"],
    }


def run_benchmark(input_path: Path = _DEFAULT_INPUT, config: STDBSCANConfig | None = None) -> list[dict[str, Any]]:
    """Run the benchmark suite and return a list of per-scenario results."""
    config = config or STDBSCANConfig()
    full_df = load_clean_detections(input_path)
    full_df = full_df.sort_values("acq_datetime").reset_index(drop=True)

    results: list[dict[str, Any]] = []

    for n in (10_000, 50_000, 200_000):
        subset = full_df.iloc[:n]
        results.append(_run_one(f"chronological_prefix_{n}", subset, config))

    in_bbox = (
        full_df["latitude"].between(_DENSE_REGION_BBOX["lat_min"], _DENSE_REGION_BBOX["lat_max"])
        & full_df["longitude"].between(_DENSE_REGION_BBOX["lon_min"], _DENSE_REGION_BBOX["lon_max"])
        & full_df["month"].isin(_DENSE_MONTHS)
    )
    dense_subset = full_df.loc[in_bbox]
    results.append(_run_one("dense_stubble_burning_region_oct_nov", dense_subset, config))

    return results


def main() -> int:
    results = run_benchmark()
    _DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with _DEFAULT_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nWrote benchmark results -> {_DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
