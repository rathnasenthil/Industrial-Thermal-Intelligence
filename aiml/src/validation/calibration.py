"""Descriptive evidence-strength analysis (NOT probability calibration)."""

from __future__ import annotations

from typing import Any, Sequence

from src.validation.metrics import evidence_strength_precision
from src.validation.validation_schema import not_evaluated_block


def analyze_calibration_proxy(
    labels: Sequence[str],
    strengths: Sequence[str | None],
    candidates: Sequence[str | None],
) -> dict[str, Any]:
    """Evaluate observed precision by evidence_strength.

    Explicitly does NOT convert industrial_evidence_score into a probability.
    """
    if not labels:
        block = not_evaluated_block("No labels for evidence-strength analysis.")
        block["note"] = (
            "I.7 ordinal scores are not probabilities; no Platt/isotonic calibration is performed."
        )
        return block
    result = evidence_strength_precision(labels, strengths, candidates)
    result["note"] = (
        "Descriptive observed precision by evidence_strength only. "
        "industrial_evidence_score is an ordinal engineering score, not an industrial_probability."
    )
    return result
