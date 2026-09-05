"""Error analysis for independent validation matches."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.validation.config import LABEL_INDUSTRIAL, MATCHED
from src.validation.metrics import map_candidate_to_binary


def analyze_errors(matched: pd.DataFrame, *, max_examples: int = 5) -> dict[str, Any]:
    """Categorize errors among independent MATCHED records."""
    if matched.empty:
        return {
            "metric_status": "NOT_EVALUATED",
            "reason": "No matched records for error analysis.",
            "categories": {},
        }

    df = matched[
        (matched["validation_match_status"] == MATCHED)
        & (matched["validation_source_independent"].fillna(False).astype(bool))
    ].copy()
    if df.empty:
        return {
            "metric_status": "NOT_EVALUATED",
            "reason": "No independent MATCHED records for error analysis.",
            "categories": {},
        }

    categories: dict[str, list[str]] = {
        "FALSE_INDUSTRIAL": [],
        "MISSED_INDUSTRIAL": [],
        "FALSE_NON_INDUSTRIAL": [],
        "ABSTENTION_ON_INDUSTRIAL": [],
        "ABSTENTION_ON_NON_INDUSTRIAL": [],
        "AMBIGUOUS_OR_UNKNOWN_LABEL": [],
        "MULTIPLE_POSSIBLE_MATCHES": [],
    }

    # Also count multiple matches from full matched table
    multi = matched[matched["validation_match_status"] == "MULTIPLE_POSSIBLE_MATCHES"]
    categories["MULTIPLE_POSSIBLE_MATCHES"] = (
        multi["validation_id"].astype(str).head(max_examples).tolist()
    )

    for _, row in df.iterrows():
        lab = str(row.get("reference_label_normalized"))
        cand = row.get("source_intelligence_candidate")
        vid = str(row.get("validation_id"))
        eid = str(row.get("event_id"))
        key = f"{vid}:{eid}"
        if lab in {"AMBIGUOUS", "UNKNOWN"}:
            categories["AMBIGUOUS_OR_UNKNOWN_LABEL"].append(key)
            continue
        pred = map_candidate_to_binary(cand, mode="strict")
        if lab == LABEL_INDUSTRIAL:
            if pred is None:
                categories["ABSTENTION_ON_INDUSTRIAL"].append(key)
            elif pred == LABEL_INDUSTRIAL:
                pass
            else:
                categories["MISSED_INDUSTRIAL"].append(key)
        else:
            if pred is None:
                categories["ABSTENTION_ON_NON_INDUSTRIAL"].append(key)
            elif pred == LABEL_INDUSTRIAL:
                categories["FALSE_INDUSTRIAL"].append(key)
            else:
                # correct non-industrial — optionally track false non-industrial only if mislabeled industrial predicted as non? skip
                pass

    summary = {
        name: {"count": len(ids), "examples": ids[:max_examples]} for name, ids in categories.items()
    }
    return {"metric_status": "EVALUATED", "categories": summary}
