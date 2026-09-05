"""
End-to-end Independent Validation pipeline (GIFT Stage V).

Read-only with respect to G→I.7 intelligence outputs.
Never fabricates labels or metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.validation.ablation import run_ablation
from src.validation.calibration import analyze_calibration_proxy
from src.validation.config import (
    MATCHED,
    METRIC_NOT_EVALUATED,
    STATUS_EVALUATED,
    STATUS_UNAVAILABLE,
    ValidationConfig,
)
from src.validation.dataset_audit import audit_validation_dataset
from src.validation.error_analysis import analyze_errors
from src.validation.event_matching import match_references_to_events
from src.validation.leakage_audit import audit_leakage
from src.validation.metrics import evaluate_binary, multiclass_metrics
from src.validation.threshold_analysis import analyze_thresholds
from src.validation.validation_loader import discover_validation_paths, load_validation_dataset
from src.validation.validation_report import build_validation_report, save_json
from src.validation.validation_schema import empty_matches_frame, not_evaluated_block


@dataclass
class ValidationResult:
    matches_df: pd.DataFrame
    metrics: dict[str, Any]
    report: dict[str, Any]


def load_events(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Events file not found: {p}")
    return pd.read_csv(p)


def run_validation(
    events_df: pd.DataFrame | None = None,
    config: ValidationConfig | None = None,
    *,
    references_df: pd.DataFrame | None = None,
    validation_path: str | Path | None = None,
) -> ValidationResult:
    """Run Stage V independent validation.

    If no independent dataset is available, returns VALIDATION_DATA_UNAVAILABLE
    with metric_status=NOT_EVALUATED (no fabricated scores).
    """
    config = config or ValidationConfig()
    start = time.perf_counter()
    warnings: list[str] = []

    if events_df is None:
        events_df = load_events(config.events_path)

    discovered = discover_validation_paths(config)
    dataset_paths = [str(p) for p in discovered]
    if validation_path is not None:
        dataset_paths = [str(validation_path)] + [p for p in dataset_paths if p != str(validation_path)]

    references = references_df
    load_meta: dict[str, Any] = {"available": False, "warnings": []}
    if references is None:
        path_to_load = Path(validation_path) if validation_path is not None else (
            discovered[0] if discovered else None
        )
        references, load_meta = load_validation_dataset(path_to_load, config)
        warnings.extend(load_meta.get("warnings") or [])
    else:
        load_meta = {"available": not references.empty, "path": "<in-memory>", "record_count": len(references)}

    dataset_audit = audit_validation_dataset(references)
    leakage = audit_leakage(references)

    available = bool(load_meta.get("available")) and not references.empty
    independent_ok = bool(leakage.get("independent_validation_confirmed"))

    if not available:
        elapsed = time.perf_counter() - start
        metrics = not_evaluated_block(
            "No independent validation dataset found under data/external, data/raw, "
            "or configured validation_path. Framework is implemented; evaluation was not performed."
        )
        report = build_validation_report(
            config=config,
            status=STATUS_UNAVAILABLE,
            validation_dataset_available=False,
            independent_validation_confirmed=False,
            dataset_paths=dataset_paths,
            dataset_audit=dataset_audit,
            match_stats={
                "matched": 0,
                "multiple_possible_matches": 0,
                "no_event_match": 0,
                "invalid_reference": 0,
            },
            leakage_audit=leakage,
            metrics=metrics,
            coverage_metrics={"evaluable_record_count": 0, "coverage": None, "metric_status": METRIC_NOT_EVALUATED},
            abstention_metrics={"abstention_rate": None, "metric_status": METRIC_NOT_EVALUATED},
            ablation_results={"metric_status": METRIC_NOT_EVALUATED, "reason": "Validation data unavailable."},
            threshold_analysis={"metric_status": METRIC_NOT_EVALUATED, "reason": "Validation data unavailable."},
            error_analysis={"metric_status": METRIC_NOT_EVALUATED, "reason": "Validation data unavailable."},
            calibration_proxy={"metric_status": METRIC_NOT_EVALUATED, "reason": "Validation data unavailable."},
            subgroup_analysis={"metric_status": METRIC_NOT_EVALUATED, "reason": "Validation data unavailable."},
            processing_seconds=elapsed,
            warnings=warnings
            + [
                "VALIDATION_DATA_UNAVAILABLE: no independent reference labels present.",
                "NO VALIDATED PERFORMANCE CLAIM IS MADE.",
            ],
        )
        return ValidationResult(matches_df=empty_matches_frame(), metrics=metrics, report=report)

    # Exclude non-independent records from primary evaluation
    primary_refs = references[references["validation_source_independent"].fillna(False).astype(bool)].copy()
    if primary_refs.empty:
        warnings.append(
            "Validation file present but no records satisfy independence criteria; "
            "excluded from primary evaluation."
        )
        elapsed = time.perf_counter() - start
        metrics = not_evaluated_block(
            "Validation references exist but none are confirmed independent."
        )
        report = build_validation_report(
            config=config,
            status=STATUS_UNAVAILABLE,
            validation_dataset_available=True,
            independent_validation_confirmed=False,
            dataset_paths=dataset_paths or [str(load_meta.get("path"))],
            dataset_audit=dataset_audit,
            match_stats={"matched": 0, "multiple_possible_matches": 0, "no_event_match": 0, "invalid_reference": 0},
            leakage_audit=leakage,
            metrics=metrics,
            coverage_metrics={"evaluable_record_count": 0, "coverage": None, "metric_status": METRIC_NOT_EVALUATED},
            abstention_metrics={"abstention_rate": None, "metric_status": METRIC_NOT_EVALUATED},
            ablation_results={"metric_status": METRIC_NOT_EVALUATED, "reason": "No independent records."},
            threshold_analysis={"metric_status": METRIC_NOT_EVALUATED, "reason": "No independent records."},
            error_analysis={"metric_status": METRIC_NOT_EVALUATED, "reason": "No independent records."},
            calibration_proxy={"metric_status": METRIC_NOT_EVALUATED, "reason": "No independent records."},
            subgroup_analysis={"metric_status": METRIC_NOT_EVALUATED, "reason": "No independent records."},
            processing_seconds=elapsed,
            warnings=warnings + ["NO VALIDATED PERFORMANCE CLAIM IS MADE."],
        )
        return ValidationResult(matches_df=empty_matches_frame(), metrics=metrics, report=report)

    matches, match_stats = match_references_to_events(primary_refs, events_df, config)
    leakage = audit_leakage(primary_refs, matches)

    eval_df = matches[
        (matches["validation_match_status"] == MATCHED)
        & (matches["validation_source_independent"].fillna(False).astype(bool))
    ].copy()
    # Drop AMBIGUOUS/UNKNOWN from binary primary metrics but keep in audit
    binary_df = eval_df[~eval_df["reference_label_normalized"].isin(["AMBIGUOUS", "UNKNOWN"])].copy()

    labels = binary_df["reference_label_normalized"].astype(str).tolist()
    candidates = binary_df["source_intelligence_candidate"].tolist()
    strengths = binary_df["evidence_strength"].tolist()
    scores = binary_df["industrial_evidence_score"].tolist()

    strict = evaluate_binary(labels, candidates, mode="strict")
    inclusive = evaluate_binary(labels, candidates, mode="inclusive")
    multi = multiclass_metrics(
        eval_df["reference_label_normalized"].astype(str).tolist(),
        # For multiclass, map abstention candidates to ABSTENTION label to avoid forcing
        [
            ("ABSTENTION" if c in {"INSUFFICIENT_EVIDENCE", "AMBIGUOUS_EVIDENCE", "MIXED_OR_CONFLICTING"} else str(c))
            for c in eval_df["source_intelligence_candidate"].tolist()
        ],
    )

    metrics = {
        "metric_status": strict.get("metric_status", METRIC_NOT_EVALUATED),
        "strict": strict,
        "inclusive": inclusive,
        "multiclass": multi,
        "primary": strict,
    }

    coverage_metrics = {
        "metric_status": strict.get("metric_status", METRIC_NOT_EVALUATED),
        "evaluable_record_count": int(len(binary_df)),
        "classified_count": int(strict.get("sample_count") or 0),
        "coverage": strict.get("coverage"),
        "matched_independent_count": int(len(eval_df)),
    }
    abstention_metrics = {
        "metric_status": strict.get("metric_status", METRIC_NOT_EVALUATED),
        "abstained_count": int(strict.get("abstained_count") or 0),
        "abstention_rate": strict.get("abstention_rate"),
    }

    ablation_results = run_ablation(matches, events_df)
    threshold_results = analyze_thresholds(labels, candidates, strengths, scores)
    error_results = analyze_errors(matches)
    calibration_proxy = analyze_calibration_proxy(labels, strengths, candidates)
    subgroup_analysis = {
        "metric_status": METRIC_NOT_EVALUATED,
        "reason": (
            f"Subgroup metrics require at least {config.min_subgroup_count} samples per group; "
            "skipped or limited unless sample sizes permit."
        ),
    }

    elapsed = time.perf_counter() - start
    status = STATUS_EVALUATED if metrics["metric_status"] == "EVALUATED" else STATUS_UNAVAILABLE
    report = build_validation_report(
        config=config,
        status=status,
        validation_dataset_available=True,
        independent_validation_confirmed=independent_ok,
        dataset_paths=dataset_paths or [str(load_meta.get("path"))],
        dataset_audit=dataset_audit,
        match_stats=match_stats,
        leakage_audit=leakage,
        metrics=metrics,
        coverage_metrics=coverage_metrics,
        abstention_metrics=abstention_metrics,
        ablation_results=ablation_results,
        threshold_analysis=threshold_results,
        error_analysis=error_results,
        calibration_proxy=calibration_proxy,
        subgroup_analysis=subgroup_analysis,
        processing_seconds=elapsed,
        warnings=warnings,
    )
    return ValidationResult(matches_df=matches, metrics=metrics, report=report)


def save_outputs(
    result: ValidationResult,
    matches_path: str | Path,
    metrics_path: str | Path,
    report_path: str | Path,
) -> None:
    matches_path = Path(matches_path)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    result.matches_df.to_csv(matches_path, index=False, na_rep="")
    save_json(result.metrics, metrics_path)
    save_json(result.report, report_path)
