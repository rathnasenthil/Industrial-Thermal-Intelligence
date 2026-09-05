"""Synthetic independent validation fixtures for Stage V tests (NOT production labels)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_synthetic_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "EVT_1",
                "centroid_latitude": 28.01,
                "centroid_longitude": 77.01,
                "event_start": "2023-01-01T00:00:00+00:00",
                "event_end": "2023-01-01T12:00:00+00:00",
                "facility_association_method": "WITHIN_FACILITY",
                "facility_attribution_confidence": "HIGH",
                "facility_type": "POWER_PLANT",
                "persistence_label": "PERSISTENT",
                "baseline_history_status": "ESTABLISHED_BASELINE",
                "anomaly_status": "NORMAL",
                "source_intelligence_candidate": "INDUSTRIAL_ACTIVITY_CANDIDATE",
                "evidence_strength": "STRONG",
                "industrial_evidence_score": 10,
            },
            {
                "event_id": "EVT_2",
                "centroid_latitude": 20.0,
                "centroid_longitude": 78.0,
                "event_start": "2023-02-01T00:00:00+00:00",
                "event_end": "2023-02-01T06:00:00+00:00",
                "facility_association_method": "NO_FACILITY_ASSOCIATION",
                "facility_attribution_confidence": "NONE",
                "facility_type": None,
                "persistence_label": "SHORT_LIVED",
                "baseline_history_status": "NOT_APPLICABLE",
                "anomaly_status": "INSUFFICIENT_HISTORY",
                "source_intelligence_candidate": "INSUFFICIENT_EVIDENCE",
                "evidence_strength": "NONE",
                "industrial_evidence_score": 0,
            },
            {
                "event_id": "EVT_3",
                "centroid_latitude": 28.02,
                "centroid_longitude": 77.02,
                "event_start": "2023-01-01T01:00:00+00:00",
                "event_end": "2023-01-01T05:00:00+00:00",
                "facility_association_method": "NEAR_FACILITY",
                "facility_attribution_confidence": "MEDIUM",
                "facility_type": "MINE",
                "persistence_label": "PERSISTENT",
                "baseline_history_status": "LIMITED_HISTORY",
                "anomaly_status": "ELEVATED",
                "source_intelligence_candidate": "POSSIBLE_INDUSTRIAL_ACTIVITY",
                "evidence_strength": "MODERATE",
                "industrial_evidence_score": 7,
            },
        ]
    )


def make_independent_references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "validation_id": "VAL_IND_1",
                "reference_label_raw": "industrial_fire",
                "reference_source": "manual_curated_independent_review",
                "reference_date": "2023-01-01T03:00:00+00:00",
                "reference_latitude": 28.011,
                "reference_longitude": 77.011,
                "validation_source_independent": True,
                "validation_label_verified": True,
            },
            {
                "validation_id": "VAL_NAT_1",
                "reference_label_raw": "wildfire",
                "reference_source": "official_incident_independent",
                "reference_date": "2023-02-01T02:00:00+00:00",
                "reference_latitude": 20.001,
                "reference_longitude": 78.001,
                "validation_source_independent": True,
                "validation_label_verified": True,
            },
            {
                "validation_id": "VAL_FAR",
                "reference_label_raw": "industrial",
                "reference_source": "manual_curated_independent_review",
                "reference_date": "2023-01-01T03:00:00+00:00",
                "reference_latitude": 10.0,
                "reference_longitude": 70.0,
                "validation_source_independent": True,
                "validation_label_verified": True,
            },
            {
                "validation_id": "VAL_CIRCULAR",
                "reference_label_raw": "industrial",
                "reference_source": "i7_candidate_export",
                "reference_date": "2023-01-01T03:00:00+00:00",
                "reference_latitude": 28.01,
                "reference_longitude": 77.01,
                "validation_source_independent": False,
                "validation_label_verified": False,
            },
        ]
    )


def write_independent_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_independent_references().to_csv(path, index=False)
    return path
