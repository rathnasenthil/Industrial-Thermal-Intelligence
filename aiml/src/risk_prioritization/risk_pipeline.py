"""
End-to-end Decision & Risk Prioritization pipeline (GIFT Stage VI).

Strictly downstream of G→I.7. Does not modify prior-stage fields or logic.
Does not create probabilities or claim validated risk accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.risk_prioritization.anomaly_priority import compute_anomaly_priority
from src.risk_prioritization.config import RiskPrioritizationConfig
from src.risk_prioritization.facility_criticality import compute_facility_context
from src.risk_prioritization.persistence_priority import compute_persistence_priority
from src.risk_prioritization.priority_explanation import build_priority_explanations
from src.risk_prioritization.priority_scoring import aggregate_priority
from src.risk_prioritization.risk_report import build_risk_report
from src.risk_prioritization.risk_schema import I7_IMMUTABLE_COLUMNS, RISK_APPEND_COLUMNS
from src.risk_prioritization.thermal_severity import compute_thermal_severity
from src.risk_prioritization.uncertainty import compute_uncertainty


@dataclass
class RiskResult:
    events_df: pd.DataFrame
    report: dict[str, Any]


def load_events(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Events file not found: {p}")
    df = pd.read_csv(p)
    if "event_id" not in df.columns:
        raise ValueError("Events table missing event_id.")
    return df


def _sanity_checks(output_df: pd.DataFrame) -> dict[str, Any]:
    """Detect pathological single-feature domination (report-only, not auto-fix)."""
    checks: dict[str, Any] = {"status": "OK", "flags": []}

    def flag(name: str, cond: bool, detail: str) -> None:
        checks[name] = {"triggered": bool(cond), "detail": detail}
        if cond:
            checks["flags"].append(name)
            checks["status"] = "ATTENTION"

    if "facility_association_method" in output_df.columns:
        within = output_df[output_df["facility_association_method"] == "WITHIN_FACILITY"]
        if len(within):
            frac = (within["investigation_priority"] == "CRITICAL").mean()
            flag(
                "within_facility_all_critical",
                frac > 0.95,
                f"CRITICAL fraction among WITHIN_FACILITY={frac:.3f}",
            )
        no_fac = output_df[output_df["facility_association_method"] == "NO_FACILITY_ASSOCIATION"]
        if len(no_fac):
            frac_low = (no_fac["investigation_priority"] == "LOW").mean()
            flag(
                "no_facility_all_low",
                frac_low >= 1.0,
                f"LOW fraction among NO_FACILITY={frac_low:.3f} (fail only if every event is LOW)",
            )

    if "persistence_label" in output_df.columns:
        pers = output_df[output_df["persistence_label"] == "PERSISTENT"]
        if len(pers):
            frac = (pers["investigation_priority"] == "CRITICAL").mean()
            flag(
                "persistent_all_critical",
                frac > 0.95,
                f"CRITICAL fraction among PERSISTENT={frac:.3f}",
            )

    # Missing STA should not force LOW
    if "sta_domain_available" in output_df.columns:
        missing_sta = output_df[~output_df["sta_domain_available"].fillna(False).astype(bool)]
        if len(missing_sta):
            frac = (missing_sta["investigation_priority"] == "LOW").mean()
            flag(
                "missing_sta_all_low",
                frac > 0.98,
                f"LOW fraction when STA unavailable={frac:.3f}",
            )

    # Near-constant risk score
    if output_df["risk_score"].nunique(dropna=False) <= 3:
        flag("risk_score_near_constant", True, f"unique risk_score values={output_df['risk_score'].nunique()}")
    else:
        flag("risk_score_near_constant", False, f"unique risk_score values={output_df['risk_score'].nunique()}")

    return checks


def run_risk_prioritization(
    events_df: pd.DataFrame,
    config: RiskPrioritizationConfig | None = None,
    *,
    events_input_path: str = "<in-memory>",
    output_path: str = "data/processed/thermal_events_with_risk_prioritization.csv",
) -> RiskResult:
    config = config or RiskPrioritizationConfig()
    start = time.perf_counter()
    working = events_df.copy()
    original_ids = set(working["event_id"].astype(str))
    warnings: list[str] = [
        "Investigation priority is a deterministic decision-support score, not a probability of industrial fire.",
        "Risk prioritization has not been independently validated because Stage V currently has no independent reference dataset.",
    ]

    thermal = compute_thermal_severity(working, config)
    persistence = compute_persistence_priority(working, config)
    anomaly = compute_anomaly_priority(working, config)
    facility = compute_facility_context(working, config)
    uncertainty = compute_uncertainty(working, config)
    scored = aggregate_priority(working, thermal, persistence, anomaly, facility, uncertainty, config)
    explained = build_priority_explanations(scored, config)

    fused = scored.merge(explained, on="event_id", how="left")
    fused = fused.set_index("event_id").reindex(working["event_id"].astype(str)).reset_index()
    fused["risk_scoring_version"] = config.scoring_version

    for col in RISK_APPEND_COLUMNS:
        working[col] = fused[col].to_numpy()

    working = working.sort_values("event_id", kind="mergesort").reset_index(drop=True)

    if set(working["event_id"].astype(str)) != original_ids:
        raise RuntimeError("Stage VI must preserve every input event_id.")
    if working["event_id"].duplicated().any():
        raise RuntimeError("Stage VI output event_id must be unique.")
    if working["risk_score"].min() < 0 or working["risk_score"].max() > 100:
        raise RuntimeError("risk_score must remain within [0, 100].")

    sanity = _sanity_checks(working)
    if sanity.get("flags"):
        warnings.append(f"Sanity attention flags: {', '.join(sanity['flags'])}")

    elapsed = time.perf_counter() - start
    report = build_risk_report(
        config=config,
        events_input_path=events_input_path,
        output_path=output_path,
        output_df=working,
        processing_seconds=elapsed,
        warnings=warnings,
        sanity=sanity,
    )
    # Attach immutability note for I.7 fields present
    report["i7_columns_preserved"] = [c for c in I7_IMMUTABLE_COLUMNS if c in working.columns]
    return RiskResult(events_df=working, report=report)


def save_outputs(result: RiskResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.events_df.to_csv(path, index=False, na_rep="")
    return path
