"""Stage V validation config/schema/loader/normalization tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.config import STATUS_UNAVAILABLE, ValidationConfig
from src.validation.label_normalization import normalize_label, normalize_reference_labels
from src.validation.validation_loader import assess_independence, load_validation_dataset
from src.validation.validation_schema import CANONICAL_COLUMNS, not_evaluated_block
from tests.fixtures.validation.make_fixtures import write_independent_csv


def test_config_defaults() -> None:
    cfg = ValidationConfig()
    assert cfg.spatial_tolerance_km == 5.0
    assert cfg.temporal_tolerance_hours == 72.0
    assert cfg.require_independent_source is True


def test_schema_and_not_evaluated_block() -> None:
    assert "validation_id" in CANONICAL_COLUMNS
    block = not_evaluated_block("missing data")
    assert block["metric_status"] == "NOT_EVALUATED"
    assert block["precision"] is None
    assert block["sample_count"] == 0


def test_label_normalization() -> None:
    assert normalize_label("industrial_fire") == "INDUSTRIAL"
    assert normalize_label("wildfire") == "NATURAL"
    assert normalize_label("stubble") == "AGRICULTURAL"
    assert normalize_label(None) == "UNKNOWN"
    df = normalize_reference_labels(pd.DataFrame({"reference_label_raw": ["wildfire", None]}))
    assert list(df["reference_label_normalized"]) == ["NATURAL", "UNKNOWN"]


def test_independence_requirement() -> None:
    cfg = ValidationConfig()
    assert assess_independence("manual_curated_independent_review", cfg) is True
    assert assess_independence("i2_facility_association", cfg) is False
    assert assess_independence("source_intelligence_candidate", cfg) is False
    assert assess_independence("unknown_source", cfg) is False


def test_loader_missing_and_present(tmp_path: Path) -> None:
    cfg = ValidationConfig()
    empty, meta = load_validation_dataset(tmp_path / "missing.csv", cfg)
    assert empty.empty
    assert meta["available"] is False

    path = write_independent_csv(tmp_path / "validation_labels.csv")
    df, meta2 = load_validation_dataset(path, cfg)
    assert meta2["available"] is True
    assert len(df) == 4
    assert "reference_label_normalized" in df.columns
    assert df["validation_source_independent"].dtype == bool or df["validation_source_independent"].isin([True, False]).all()


def test_status_constant() -> None:
    assert STATUS_UNAVAILABLE == "VALIDATION_DATA_UNAVAILABLE"
