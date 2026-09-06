"""
Incremental Stage I.3 facility thermal fingerprinting (AIML realtime adapter).

I.3 is a **descriptive** facility thermal baseline — not anomaly detection
and not source / industrial-fire classification.

Why realtime I.3 cannot replay the full batch pipeline
------------------------------------------------------
Batch ``run_facility_fingerprinting()`` builds fingerprints for every
facility in the Stage I.1 universe. On each NRT poll only the facility
(or facilities) touched by Phase 5 association change. Re-running the
batch entrypoint would re-touch ~100k+ unrelated facilities.

This adapter calls the same batch builders
(``build_facility_fingerprints``, ``build_monthly_profile``) on a
**one-facility** frame so formulas, thresholds, and robust stats remain
identical.

Confirmed associations only
---------------------------
Primary statistics use events with non-null ``facility_id`` (methods
WITHIN / INTERSECTS / NEAR). AMBIGUOUS events never increase
``event_count``; they only affect ``ambiguous_candidate_opportunity_count``
via reconstructed ``candidate_facility_ids``. NO_FACILITY_ASSOCIATION
events are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from src.fingerprinting.facility_fingerprint import (
    OUTPUT_COLUMNS,
    REQUIRED_EVENT_COLUMNS,
    build_facility_fingerprints,
)
from src.fingerprinting.fingerprint_config import DEFAULT_CONFIG, FingerprintConfig
from src.fingerprinting.monthly_profile import OUTPUT_COLUMNS as MONTHLY_COLUMNS
from src.fingerprinting.monthly_profile import build_monthly_profile

CONFIRMED_ASSOCIATION_METHODS: frozenset[str] = frozenset(
    {"WITHIN_FACILITY", "INTERSECTS_FACILITY", "NEAR_FACILITY"}
)


@dataclass(frozen=True)
class FacilityFingerprintResult:
    """One facility fingerprint + its sparse monthly profile."""

    fingerprint: dict[str, Any]
    monthly_profile: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": dict(self.fingerprint),
            "monthly_profile": [dict(r) for r in self.monthly_profile],
        }


def _series_to_plain(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in OUTPUT_COLUMNS:
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            out[key] = None
        elif hasattr(val, "to_pydatetime"):
            out[key] = val.to_pydatetime() if pd.notna(val) else None
        elif pd.isna(val):
            out[key] = None
        else:
            # numpy scalars → Python
            try:
                if hasattr(val, "item"):
                    out[key] = val.item()
                else:
                    out[key] = val
            except Exception:
                out[key] = val
    return out


def process_facility_fingerprint(
    facility: Mapping[str, Any],
    events_df: pd.DataFrame,
    *,
    config: Optional[FingerprintConfig] = None,
) -> FacilityFingerprintResult:
    """
    Build I.3 fingerprint + monthly profile for **one** facility.

    Args:
        facility: Mapping with ``facility_id``, ``facility_name``,
            ``facility_type``.
        events_df: Event rows for this facility's confirmed associations
            and (optionally) AMBIGUOUS events that list it as a candidate.
            Must include ``REQUIRED_EVENT_COLUMNS`` (provide
            ``candidate_facility_ids`` reconstructed from candidates;
            do not invent ThermalEvent columns).
        config: Defaults to batch ``DEFAULT_CONFIG``.

    Returns:
        FacilityFingerprintResult with batch-identical semantics.
    """
    cfg = config or DEFAULT_CONFIG
    facility_id = str(facility["facility_id"])
    facilities_df = pd.DataFrame(
        [
            {
                "facility_id": facility_id,
                "facility_name": facility.get("facility_name"),
                "facility_type": facility.get("facility_type"),
            }
        ]
    )

    # Ensure required columns exist (empty frame still valid).
    work = events_df.copy() if events_df is not None else pd.DataFrame()
    if work.empty:
        work = pd.DataFrame(columns=list(REQUIRED_EVENT_COLUMNS))
    else:
        for col in REQUIRED_EVENT_COLUMNS:
            if col not in work.columns:
                work[col] = None if col != "candidate_facility_ids" else ""

    fingerprints = build_facility_fingerprints(work, facilities_df, cfg)
    assert len(fingerprints) == 1
    assert fingerprints.iloc[0]["facility_id"] == facility_id

    monthly = build_monthly_profile(work)
    if not monthly.empty:
        monthly = monthly.loc[monthly["facility_id"] == facility_id].reset_index(drop=True)
    else:
        monthly = pd.DataFrame(columns=list(MONTHLY_COLUMNS))

    monthly_rows: list[dict[str, Any]] = []
    for _, mrow in monthly.iterrows():
        monthly_rows.append(
            {
                "facility_id": str(mrow["facility_id"]),
                "month": int(mrow["month"]),
                "event_count": int(mrow["event_count"]),
                "detection_count": int(mrow["detection_count"]),
                "event_fraction": float(mrow["event_fraction"]),
            }
        )

    return FacilityFingerprintResult(
        fingerprint=_series_to_plain(fingerprints.iloc[0]),
        monthly_profile=monthly_rows,
    )
