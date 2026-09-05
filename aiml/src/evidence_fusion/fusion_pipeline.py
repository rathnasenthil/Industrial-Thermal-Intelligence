"""
End-to-end Evidence Fusion / Source Intelligence pipeline (GIFT Stage I.7).

Appends fusion fields without modifying G→I.6 logic or prior-stage fields.
Does not train ML, create pseudo-labels, or claim ground truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evidence_fusion.availability import build_availability_profile
from src.evidence_fusion.candidate_interpretation import interpret_candidates
from src.evidence_fusion.config import EvidenceFusionConfig
from src.evidence_fusion.conflicts import detect_evidence_conflicts
from src.evidence_fusion.environmental_evidence import extract_environmental_evidence
from src.evidence_fusion.fusion_report import build_fusion_report
from src.evidence_fusion.fusion_schema import (
    FORBIDDEN_SUBSTRINGS,
    FUSION_COLUMNS,
    I4_IMMUTABLE_COLUMNS,
    I5_IMMUTABLE_COLUMNS,
)
from src.evidence_fusion.infrastructure_evidence import extract_infrastructure_evidence
from src.evidence_fusion.sta_fusion_evidence import extract_sta_evidence
from src.evidence_fusion.temporal_evidence import extract_temporal_evidence


@dataclass
class FusionResult:
    events_df: pd.DataFrame
    report: dict[str, Any]


def load_events(path: str | Path) -> pd.DataFrame:
    events_path = Path(path)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    df = pd.read_csv(events_path)
    if "event_id" not in df.columns:
        raise ValueError(f"'{events_path}' missing required column: event_id.")
    return df


def run_evidence_fusion(
    events_df: pd.DataFrame,
    config: EvidenceFusionConfig | None = None,
    *,
    events_input_path: str = "<in-memory>",
    output_path: str = "data/processed/thermal_events_with_evidence_fusion.csv",
) -> FusionResult:
    """Run Stage I.7 over an already-loaded events table."""
    config = config or EvidenceFusionConfig()
    start = time.perf_counter()
    working = events_df.copy()
    original_ids = set(working["event_id"].astype(str))
    warnings: list[str] = []

    if "sta_association_status" not in working.columns:
        warnings.append(
            "I.5 STA columns absent on input. STA domain marked unavailable for all "
            "events (missing != anti-industrial)."
        )
    env_cols = [
        "landcover_available",
        "vegetation_context_available",
        "builtup_context_available",
        "water_context_available",
        "agriculture_context_available",
        "satellite_context_available",
    ]
    if not any(c in working.columns for c in env_cols):
        warnings.append("I.6 environmental availability columns absent on input.")
    elif all(
        (not working[c].fillna(False).astype(bool).any())
        for c in env_cols
        if c in working.columns
    ):
        warnings.append(
            "No environmental datasets were available in I.6 for this run. "
            "Environmental domain remains unavailable (missing != negative evidence)."
        )

    temporal = extract_temporal_evidence(working)
    infrastructure = extract_infrastructure_evidence(working)
    sta = extract_sta_evidence(working)
    environmental = extract_environmental_evidence(working)
    availability = build_availability_profile(temporal, infrastructure, sta, environmental)
    conflicts = detect_evidence_conflicts(infrastructure, sta, environmental)
    interpretation = interpret_candidates(
        working,
        temporal,
        infrastructure,
        sta,
        environmental,
        availability,
        conflicts,
        config,
    )

    fused = temporal
    for frame in (infrastructure, sta, environmental, availability, conflicts, interpretation):
        fused = fused.merge(frame, on="event_id", how="left")

    fused = fused.set_index("event_id").reindex(working["event_id"].astype(str)).reset_index()

    for col in FUSION_COLUMNS:
        working[col] = fused[col].to_numpy()

    # Deterministic ordering.
    working = working.sort_values("event_id", kind="mergesort").reset_index(drop=True)

    if set(working["event_id"].astype(str)) != original_ids:
        raise RuntimeError("I.7 must preserve every input event_id.")
    if working["event_id"].duplicated().any():
        raise RuntimeError("I.7 output event_id must be unique.")
    if not working["candidate_is_ground_truth"].eq(False).all():
        raise RuntimeError("candidate_is_ground_truth must be False for every event.")

    blob = " ".join(working.columns).lower()
    for term in FORBIDDEN_SUBSTRINGS:
        if term in blob:
            raise RuntimeError(f"Forbidden classification/risk column substring present: {term}")

    elapsed = time.perf_counter() - start
    report = build_fusion_report(
        config=config,
        events_input_path=events_input_path,
        output_path=output_path,
        event_count=len(working),
        output_df=working,
        processing_seconds=elapsed,
        warnings=warnings,
        domain_availability={
            "temporal": bool(temporal["temporal_evidence_available"].any()),
            "infrastructure": bool(infrastructure["infrastructure_evidence_available"].any()),
            "sta": bool(sta["sta_domain_available"].any()),
            "environmental": bool(environmental["environmental_domain_available"].any()),
        },
        i4_columns_present=[c for c in I4_IMMUTABLE_COLUMNS if c in working.columns],
        i5_columns_present=[c for c in I5_IMMUTABLE_COLUMNS if c in working.columns],
    )
    return FusionResult(events_df=working, report=report)


def save_outputs(result: FusionResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Empty strings for object nulls; leave numeric NaN as empty via na_rep.
    result.events_df.to_csv(path, index=False, na_rep="")
    return path
