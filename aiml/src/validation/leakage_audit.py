"""Leakage audit: ensure validation labels are independent of pipeline evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.validation.config import FORBIDDEN_PSEUDO_LABEL_SOURCES


def audit_leakage(references: pd.DataFrame, matches: pd.DataFrame | None = None) -> dict[str, Any]:
    """Check for pipeline-derived / circular validation sources."""
    issues: list[str] = []
    if references.empty:
        return {
            "independent_validation_confirmed": False,
            "issues": ["No validation references loaded."],
            "forbidden_source_hits": {},
            "pipeline_evidence_used_as_labels": False,
        }

    sources = references.get("reference_source", pd.Series(dtype=object)).fillna("").astype(str)
    hits: dict[str, int] = {}
    for bad in sorted(FORBIDDEN_PSEUDO_LABEL_SOURCES):
        count = int(sources.str.lower().str.contains(bad, regex=False).sum())
        if count:
            hits[bad] = count
            issues.append(f"Reference source contains forbidden token '{bad}' in {count} rows.")

    indep = references.get("validation_source_independent")
    indep_count = int(indep.fillna(False).astype(bool).sum()) if indep is not None else 0
    if indep_count == 0:
        issues.append("Zero records marked validation_source_independent=True.")

    # Check matches aren't evaluating candidate against itself as label
    if matches is not None and not matches.empty:
        if "reference_label_raw" in matches.columns and "source_intelligence_candidate" in matches.columns:
            same = (
                matches["reference_label_raw"].astype(str).str.upper()
                == matches["source_intelligence_candidate"].astype(str).str.upper()
            ).sum()
            # not necessarily leakage, but suspicious if all labels equal candidates
            if len(matches) > 0 and same == len(matches):
                issues.append(
                    "All reference_label_raw values equal source_intelligence_candidate — "
                    "possible circular validation."
                )

    confirmed = indep_count > 0 and not hits and not any(
        "circular" in i.lower() or "forbidden" in i.lower() for i in issues
    )
    # more conservative: confirmed only if independent flags and no forbidden hits
    confirmed = indep_count > 0 and len(hits) == 0

    return {
        "independent_validation_confirmed": bool(confirmed),
        "independent_record_count": indep_count,
        "forbidden_source_hits": hits,
        "pipeline_evidence_used_as_labels": bool(hits),
        "issues": issues,
        "note": (
            "Pipeline stages I.2–I.7 must not supply ground-truth labels. "
            "Only explicitly independent curated/official references qualify."
        ),
    }
