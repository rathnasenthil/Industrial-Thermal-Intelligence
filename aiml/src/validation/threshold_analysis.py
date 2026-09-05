"""Threshold / evidence-strength sensitivity analysis (descriptive, not tuning)."""

from __future__ import annotations

from typing import Any, Sequence

from src.validation.metrics import evaluate_binary, evidence_strength_precision
from src.validation.validation_schema import not_evaluated_block


def analyze_thresholds(
    labels: Sequence[str],
    candidates: Sequence[str | None],
    strengths: Sequence[str | None],
    industrial_scores: Sequence[float | None],
) -> dict[str, Any]:
    """Describe performance under documented evaluation modes / score floors.

    Does NOT optimize thresholds against the validation set.
    """
    if not labels:
        return {
            "metric_status": "NOT_EVALUATED",
            "reason": "No labels available for threshold analysis.",
            "modes": {},
            "score_floors": {},
            "strength_analysis": not_evaluated_block("No labels."),
        }

    modes = {
        "strict": evaluate_binary(labels, candidates, mode="strict"),
        "inclusive": evaluate_binary(labels, candidates, mode="inclusive"),
    }

    # Score-floor descriptive filters on already-produced industrial_evidence_score
    floors: dict[str, Any] = {}
    for floor in (0, 4, 6, 8, 10):
        filt_labels = []
        filt_cands = []
        for lab, cand, score in zip(labels, candidates, industrial_scores):
            try:
                s = float(score) if score is not None else float("nan")
            except (TypeError, ValueError):
                s = float("nan")
            if s != s:  # NaN
                continue
            if s < floor:
                # treat as abstention by passing INSUFFICIENT
                filt_labels.append(lab)
                filt_cands.append("INSUFFICIENT_EVIDENCE")
            else:
                filt_labels.append(lab)
                filt_cands.append(cand)
        floors[f"min_industrial_score_{floor}"] = evaluate_binary(
            filt_labels, filt_cands, mode="strict"
        )

    return {
        "metric_status": "EVALUATED",
        "note": (
            "Threshold analysis is descriptive under fixed engineering floors. "
            "It is not a held-out hyperparameter search and must not be read as "
            "optimized scientific thresholds."
        ),
        "modes": modes,
        "score_floors": floors,
        "strength_analysis": evidence_strength_precision(labels, strengths, candidates),
    }
