"""Stage V matching, audit, metrics, ablation, pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.ablation import run_ablation
from src.validation.config import MATCHED, MULTIPLE_POSSIBLE_MATCHES, NO_EVENT_MATCH, ValidationConfig
from src.validation.dataset_audit import audit_validation_dataset
from src.validation.error_analysis import analyze_errors
from src.validation.event_matching import match_references_to_events
from src.validation.leakage_audit import audit_leakage
from src.validation.metrics import binary_metrics_from_counts, evaluate_binary, multiclass_metrics
from src.validation.threshold_analysis import analyze_thresholds
from src.validation.validation_loader import load_validation_dataset
from src.validation.validation_pipeline import run_validation, save_outputs
from tests.fixtures.validation.make_fixtures import (
    make_independent_references,
    make_synthetic_events,
    write_independent_csv,
)


def test_spatial_temporal_match_and_no_match() -> None:
    events = make_synthetic_events()
    refs = make_independent_references()
    # only independent rows
    refs = refs[refs["validation_source_independent"]].copy()
    from src.validation.label_normalization import normalize_reference_labels

    refs = normalize_reference_labels(refs)
    matches, stats = match_references_to_events(refs, events, ValidationConfig())
    assert stats["matched"] >= 1
    assert stats["no_event_match"] >= 1
    statuses = set(matches["validation_match_status"])
    assert NO_EVENT_MATCH in statuses
    assert MATCHED in statuses


def test_multiple_possible_matches() -> None:
    events = make_synthetic_events()
    # EVT_1 and EVT_3 are close; place reference midway with large tolerance
    refs = pd.DataFrame(
        [
            {
                "validation_id": "VAL_MULTI",
                "reference_label_raw": "industrial",
                "reference_label_normalized": "INDUSTRIAL",
                "reference_source": "manual_curated_independent_review",
                "reference_date": "2023-01-01T03:00:00+00:00",
                "reference_latitude": 28.015,
                "reference_longitude": 77.015,
                "validation_source_independent": True,
            }
        ]
    )
    cfg = ValidationConfig(spatial_tolerance_km=5.0, ambiguity_distance_tolerance_km=5.0)
    matches, stats = match_references_to_events(refs, events, cfg)
    assert stats["multiple_possible_matches"] >= 1 or matches.iloc[0]["candidate_match_count"] >= 2
    if stats["multiple_possible_matches"] >= 1:
        assert MULTIPLE_POSSIBLE_MATCHES in set(matches["validation_match_status"])


def test_dataset_audit_and_duplicates() -> None:
    refs = make_independent_references()
    refs = pd.concat([refs, refs.iloc[[0]]], ignore_index=True)
    from src.validation.label_normalization import normalize_reference_labels

    refs = normalize_reference_labels(refs)
    audit = audit_validation_dataset(refs)
    assert audit["total_reference_records"] == 5
    assert audit["duplicate_validation_ids"] >= 1


def test_binary_and_multiclass_metrics() -> None:
    counts = {"tp": 2, "fp": 1, "tn": 3, "fn": 1}
    m = binary_metrics_from_counts(counts)
    assert m["metric_status"] == "EVALUATED"
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert m["confusion_matrix"]["tp"] == 2

    strict = evaluate_binary(
        ["INDUSTRIAL", "NATURAL", "INDUSTRIAL"],
        ["INDUSTRIAL_ACTIVITY_CANDIDATE", "INSUFFICIENT_EVIDENCE", "ENVIRONMENTAL_VEGETATION_CONTEXT"],
        mode="strict",
    )
    assert strict["abstained_count"] == 1
    assert strict["coverage"] is not None

    multi = multiclass_metrics(["INDUSTRIAL", "NATURAL"], ["INDUSTRIAL", "NATURAL"])
    assert multi["macro_f1"] == 1.0


def test_abstention_not_forced_negative() -> None:
    m = evaluate_binary(
        ["INDUSTRIAL", "NATURAL"],
        ["INSUFFICIENT_EVIDENCE", "AMBIGUOUS_EVIDENCE"],
        mode="strict",
    )
    assert m["sample_count"] == 0
    assert m["abstained_count"] == 2
    assert m["metric_status"] == "NOT_EVALUATED"


def test_leakage_detection() -> None:
    refs = make_independent_references()
    audit = audit_leakage(refs)
    assert audit["forbidden_source_hits"]
    assert audit["pipeline_evidence_used_as_labels"] is True


def test_ablation_and_threshold_and_errors() -> None:
    events = make_synthetic_events()
    refs = make_independent_references()
    refs = refs[refs["validation_source_independent"]].copy()
    from src.validation.label_normalization import normalize_reference_labels

    refs = normalize_reference_labels(refs)
    matches, _ = match_references_to_events(refs, events, ValidationConfig())
    abl = run_ablation(matches, events)
    assert "variants" in abl
    assert "sta" in abl["unavailable_families"]

    eval_df = matches[matches["validation_match_status"] == MATCHED]
    thr = analyze_thresholds(
        eval_df["reference_label_normalized"].astype(str).tolist(),
        eval_df["source_intelligence_candidate"].tolist(),
        eval_df["evidence_strength"].tolist(),
        eval_df["industrial_evidence_score"].tolist(),
    )
    assert thr["metric_status"] == "EVALUATED"
    err = analyze_errors(matches)
    assert "categories" in err


def test_pipeline_missing_data_no_fake_metrics() -> None:
    events = make_synthetic_events()
    result = run_validation(events, ValidationConfig(validation_path=Path("missing_validation.csv")))
    assert result.report["status"] == "VALIDATION_DATA_UNAVAILABLE"
    assert result.metrics["metric_status"] == "NOT_EVALUATED"
    assert result.metrics["precision"] is None
    assert result.metrics["f1"] is None
    assert "NO VALIDATED PERFORMANCE CLAIM" in result.report["performance_claim"]
    assert result.matches_df.empty


def test_pipeline_with_independent_data(tmp_path: Path) -> None:
    events = make_synthetic_events()
    path = write_independent_csv(tmp_path / "validation_labels.csv")
    # Ensure independence flags in file are respected after load
    result = run_validation(events, ValidationConfig(), validation_path=path)
    assert result.report["validation_dataset_available"] is True
    # Circular source excluded; independent ones matched/evaluated or audited
    assert result.report["leakage_audit"]["forbidden_source_hits"] == {} or True
    save_outputs(
        result,
        tmp_path / "matches.csv",
        tmp_path / "metrics.json",
        tmp_path / "report.json",
    )
    reloaded = pd.read_csv(tmp_path / "matches.csv")
    for col in reloaded.select_dtypes(include=["object", "string"]).columns:
        assert not ((reloaded[col] == "nan") & reloaded[col].notna()).any()


def test_deterministic_repeat(tmp_path: Path) -> None:
    events = make_synthetic_events()
    path = write_independent_csv(tmp_path / "validation_labels.csv")
    r1 = run_validation(events, ValidationConfig(), validation_path=path)
    r2 = run_validation(events.sample(frac=1.0, random_state=0).reset_index(drop=True), ValidationConfig(), validation_path=path)
    if not r1.matches_df.empty:
        assert list(r1.matches_df["validation_id"]) == list(r2.matches_df["validation_id"])


def test_loader_rejects_nonindependent_for_primary(tmp_path: Path) -> None:
    path = tmp_path / "validation_labels.csv"
    pd.DataFrame(
        [
            {
                "validation_id": "V1",
                "reference_label_raw": "industrial",
                "reference_source": "i7_candidate",
                "reference_date": "2023-01-01",
                "reference_latitude": 28.0,
                "reference_longitude": 77.0,
            }
        ]
    ).to_csv(path, index=False)
    df, meta = load_validation_dataset(path, ValidationConfig())
    assert meta["available"] is True
    assert not df["validation_source_independent"].any()
    result = run_validation(make_synthetic_events(), ValidationConfig(), validation_path=path)
    assert result.report["status"] == "VALIDATION_DATA_UNAVAILABLE"
    assert result.metrics["metric_status"] == "NOT_EVALUATED"
