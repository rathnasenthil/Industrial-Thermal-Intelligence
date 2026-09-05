"""Audit independent validation reference datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd


def audit_validation_dataset(references: pd.DataFrame) -> dict[str, Any]:
    """Produce a descriptive audit of reference records (no silent dropping)."""
    if references.empty:
        return {
            "total_reference_records": 0,
            "date_range": None,
            "geographic_coverage": None,
            "label_distribution_raw": {},
            "label_distribution_normalized": {},
            "missing_coordinates": 0,
            "missing_labels": 0,
            "duplicate_validation_ids": 0,
            "ambiguous_references": 0,
            "independent_record_count": 0,
            "non_independent_record_count": 0,
            "source_coverage": {},
        }

    lat = pd.to_numeric(references.get("reference_latitude"), errors="coerce")
    lon = pd.to_numeric(references.get("reference_longitude"), errors="coerce")
    missing_coords = int((lat.isna() | lon.isna()).sum())
    missing_labels = int(references["reference_label_raw"].isna().sum()) if "reference_label_raw" in references else len(references)
    dup_ids = int(references["validation_id"].duplicated().sum()) if "validation_id" in references else 0
    ambiguous = 0
    if "reference_label_normalized" in references:
        ambiguous = int(references["reference_label_normalized"].isin(["AMBIGUOUS", "UNKNOWN"]).sum())

    dates = pd.to_datetime(references.get("reference_date"), utc=True, errors="coerce")
    date_range = None
    if dates.notna().any():
        date_range = {
            "min": dates.min().isoformat(),
            "max": dates.max().isoformat(),
        }

    geo = None
    if lat.notna().any() and lon.notna().any():
        geo = {
            "min_latitude": float(lat.min()),
            "max_latitude": float(lat.max()),
            "min_longitude": float(lon.min()),
            "max_longitude": float(lon.max()),
        }

    indep = references.get("validation_source_independent")
    indep_count = int(indep.fillna(False).astype(bool).sum()) if indep is not None else 0

    return {
        "total_reference_records": int(len(references)),
        "date_range": date_range,
        "geographic_coverage": geo,
        "label_distribution_raw": (
            references["reference_label_raw"].value_counts(dropna=False).astype(int).to_dict()
            if "reference_label_raw" in references
            else {}
        ),
        "label_distribution_normalized": (
            references["reference_label_normalized"].value_counts(dropna=False).astype(int).to_dict()
            if "reference_label_normalized" in references
            else {}
        ),
        "missing_coordinates": missing_coords,
        "missing_labels": missing_labels,
        "duplicate_validation_ids": dup_ids,
        "ambiguous_references": ambiguous,
        "independent_record_count": indep_count,
        "non_independent_record_count": int(len(references) - indep_count),
        "source_coverage": (
            references["reference_source"].value_counts(dropna=False).astype(int).to_dict()
            if "reference_source" in references
            else {}
        ),
    }
