"""
End-to-end Thermal Event <-> Facility Association pipeline (GIFT Stage I.2).

Wires together, in order:

1. Load the Stage G/G.1 event table (immutable baseline -- never
   rewritten, never re-clustered) and the Stage I.1 normalized facility
   layer (immutable baseline -- never re-derived from the raw PBF here).
2. Parse event/facility geometry (`association_geometry`).
3. Spatial-index candidate search (`association_geometry.find_candidate_pairs`).
4. Deterministic ranking + association selection (`facility_association`).
5. Merge the selected association back onto a COPY of the original event
   table (append-only: every original column is preserved unchanged; new
   columns are only ever added, never replacing existing ones) and
   assemble a JSON report.

This stage NEVER reads Stage G/G.1 source code or re-runs ST-DBSCAN /
persistence classification, and NEVER re-ingests the OSM PBF -- it only
reads the already-produced `thermal_events*.csv` and `osm_facilities.*`
files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.infrastructure.association_config import AssociationConfig
from src.infrastructure.association_geometry import (
    build_event_geometries,
    find_candidate_pairs,
    load_facilities_geodataframe,
)
from src.infrastructure.association_report import build_association_report
from src.infrastructure.facility_association import (
    CANDIDATES_OUTPUT_COLUMNS,
    MAIN_OUTPUT_COLUMNS,
    rank_candidates,
    select_association,
)

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = ("event_id", "centroid_wkt", "footprint_wkt")


@dataclass
class AssociationResult:
    """Result of running the Stage I.2 pipeline.

    Attributes:
        events_df: The input event table with `MAIN_OUTPUT_COLUMNS`
            appended. Same row count and `event_id` set as the input --
            this stage never drops, merges or duplicates events.
        candidates_df: One row per retained (event, facility) candidate
            pair, ranked (see `facility_association.rank_candidates`).
            Every `event_id` here also appears in `events_df`; every
            `facility_id` here also appears in the input facility table.
        report: JSON-serializable Stage I.2 report.
    """

    events_df: pd.DataFrame
    candidates_df: pd.DataFrame
    report: dict[str, Any]


def load_events(path: str | Path) -> pd.DataFrame:
    """Load the Stage G/G.1 event table (read-only; never modified on disk).

    Args:
        path: Path to `thermal_events_with_persistence.csv` (preferred)
            or `thermal_events.csv`.

    Returns:
        The events DataFrame, unmodified.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If required columns are missing.
    """
    events_path = Path(path)
    if not events_path.exists():
        raise FileNotFoundError(f"Thermal events file not found: {events_path}")

    df = pd.read_csv(events_path)
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{events_path}' is missing required column(s): {missing}. "
            "Stage I.2 expects the output of src.event_formation / "
            "src.persistence."
        )
    return df


def run_facility_association(
    events_df: pd.DataFrame,
    facilities_path: str | Path,
    config: AssociationConfig,
    events_input_path: str = "<in-memory>",
) -> AssociationResult:
    """Run the full Stage I.2 pipeline over already-loaded inputs.

    Args:
        events_df: Output of `load_events` (or an equivalent DataFrame
            with `REQUIRED_EVENT_COLUMNS`).
        facilities_path: Path to the Stage I.1 facility layer (GeoJSON or
            CSV -- see `association_geometry.load_facilities_geodataframe`).
        config: `AssociationConfig`.
        events_input_path: Recorded in the report for provenance only.

    Returns:
        An `AssociationResult`.
    """
    start_time = time.perf_counter()

    facilities_gdf = load_facilities_geodataframe(facilities_path)
    events_gdf = build_event_geometries(events_df)

    pairs_df = find_candidate_pairs(events_gdf, facilities_gdf, config.association_radius_km)
    ranked_df = rank_candidates(pairs_df)
    selection_df = select_association(events_df["event_id"], ranked_df, config)

    # Invariant: append-only merge. Every original event column survives
    # unchanged; only the new MAIN_OUTPUT_COLUMNS are added.
    events_with_association = events_df.merge(selection_df, on="event_id", how="left", validate="one_to_one")
    assert len(events_with_association) == len(events_df)
    assert set(events_with_association["event_id"]) == set(events_df["event_id"])
    for col in events_df.columns:
        assert col in events_with_association.columns
    for col in MAIN_OUTPUT_COLUMNS:
        assert col in events_with_association.columns

    candidates_df = ranked_df[list(CANDIDATES_OUTPUT_COLUMNS)].copy() if not ranked_df.empty else pd.DataFrame(
        columns=list(CANDIDATES_OUTPUT_COLUMNS)
    )
    if config.max_candidates_per_event is not None and not candidates_df.empty:
        candidates_df = candidates_df.loc[candidates_df["candidate_rank"] <= config.max_candidates_per_event]

    processing_seconds = time.perf_counter() - start_time

    report = build_association_report(
        config=config,
        events_input_path=str(events_input_path),
        facilities_input_path=str(facilities_path),
        event_count=len(events_df),
        facility_count=len(facilities_gdf),
        events_with_association_df=events_with_association,
        candidates_df=candidates_df,
        processing_seconds=processing_seconds,
    )

    return AssociationResult(events_df=events_with_association, candidates_df=candidates_df, report=report)


def save_outputs(result: AssociationResult, events_output_path: str | Path, candidates_output_path: str | Path) -> None:
    """Write the Stage I.2 outputs to CSV.

    Args:
        result: Output of `run_facility_association`.
        events_output_path: Destination for
            `thermal_events_with_facility_association.csv` (all original
            event columns plus `MAIN_OUTPUT_COLUMNS`).
        candidates_output_path: Destination for
            `thermal_event_facility_candidates.csv`.
    """
    events_path = Path(events_output_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    result.events_df.to_csv(events_path, index=False)

    candidates_path = Path(candidates_output_path)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    result.candidates_df.to_csv(candidates_path, index=False)
