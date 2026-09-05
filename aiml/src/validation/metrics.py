"""Classification metrics for independent validation (no fake placeholders)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

from src.validation.config import (
    ABSTENTION_CANDIDATES,
    INCLUSIVE_POSITIVE_CANDIDATES,
    LABEL_INDUSTRIAL,
    STRICT_NEGATIVE_CANDIDATES,
    STRICT_POSITIVE_CANDIDATES,
)
from src.validation.validation_schema import not_evaluated_block


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def confusion_counts(y_true: Sequence[str], y_pred: Sequence[str], positive: str) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t == positive
        p_pos = p == positive
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif not t_pos and not p_pos:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def binary_metrics_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    bal = None
    if recall is not None and specificity is not None:
        bal = 0.5 * (recall + specificity)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "metric_status": "EVALUATED",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": bal,
        "accuracy": acc,
        "ppv": precision,
        "npv": npv,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "sample_count": int(tp + tn + fp + fn),
    }


def map_candidate_to_binary(
    candidate: str | None,
    *,
    mode: str = "strict",
) -> str | None:
    """Map I.7 candidate to INDUSTRIAL / NON_INDUSTRIAL / None(abstain).

    Documented evaluation mapping — not a claim that candidate == ground truth.
    """
    if candidate is None:
        return None
    c = str(candidate)
    if c in ABSTENTION_CANDIDATES:
        return None
    positives = STRICT_POSITIVE_CANDIDATES if mode == "strict" else INCLUSIVE_POSITIVE_CANDIDATES
    if c in positives:
        return LABEL_INDUSTRIAL
    if c in STRICT_NEGATIVE_CANDIDATES or mode == "inclusive":
        # inclusive: anything non-positive non-abstention treated as non-industrial
        if c not in positives:
            return "NON_INDUSTRIAL"
    if mode == "strict":
        # POSSIBLE and other non-listed → abstain under strict
        if c == "POSSIBLE_INDUSTRIAL_ACTIVITY":
            return None
        return "NON_INDUSTRIAL" if c not in positives else LABEL_INDUSTRIAL
    return "NON_INDUSTRIAL"


def evaluate_binary(
    y_true: Sequence[str],
    candidates: Sequence[str | None],
    *,
    mode: str = "strict",
    positive_label: str = LABEL_INDUSTRIAL,
    negative_labels: Iterable[str] = ("NATURAL", "AGRICULTURAL", "OTHER"),
) -> dict[str, Any]:
    """Evaluate I.7 candidates against independent binary-capable labels."""
    neg = set(negative_labels)
    mapped_true: list[str] = []
    mapped_pred: list[str] = []
    abstained = 0
    skipped_label = 0
    for t, c in zip(y_true, candidates):
        if t == positive_label:
            true_bin = LABEL_INDUSTRIAL
        elif t in neg:
            true_bin = "NON_INDUSTRIAL"
        else:
            skipped_label += 1
            continue
        pred = map_candidate_to_binary(c, mode=mode)
        if pred is None:
            abstained += 1
            continue
        mapped_true.append(true_bin)
        mapped_pred.append(pred)

    total_eligible = len(mapped_true) + abstained
    coverage = _safe_div(len(mapped_true), total_eligible) if total_eligible else None
    abstention_rate = _safe_div(abstained, total_eligible) if total_eligible else None

    if not mapped_true:
        block = not_evaluated_block(
            "No classified (non-abstention) pairs available for binary evaluation."
        )
        block.update(
            {
                "mode": mode,
                "coverage": coverage,
                "abstention_rate": abstention_rate,
                "abstained_count": abstained,
                "skipped_ambiguous_unknown_labels": skipped_label,
                "evaluation_mapping": {
                    "note": (
                        "I.7 candidate categories are mapped to binary INDUSTRIAL/"
                        "NON_INDUSTRIAL for evaluation only; this is not identity with ground truth."
                    ),
                    "strict_positive": sorted(STRICT_POSITIVE_CANDIDATES),
                    "inclusive_positive": sorted(INCLUSIVE_POSITIVE_CANDIDATES),
                    "abstentions": sorted(ABSTENTION_CANDIDATES),
                },
            }
        )
        return block

    counts = confusion_counts(mapped_true, mapped_pred, LABEL_INDUSTRIAL)
    metrics = binary_metrics_from_counts(counts)
    metrics.update(
        {
            "mode": mode,
            "coverage": coverage,
            "abstention_rate": abstention_rate,
            "abstained_count": abstained,
            "skipped_ambiguous_unknown_labels": skipped_label,
            "evaluation_mapping": {
                "note": (
                    "I.7 candidate categories are mapped to binary INDUSTRIAL/"
                    "NON_INDUSTRIAL for evaluation only; this is not identity with ground truth."
                ),
                "strict_positive": sorted(STRICT_POSITIVE_CANDIDATES),
                "inclusive_positive": sorted(INCLUSIVE_POSITIVE_CANDIDATES),
                "abstentions": sorted(ABSTENTION_CANDIDATES),
            },
        }
    )
    return metrics


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Macro/weighted/per-class metrics for multi-class independent labels."""
    if not y_true:
        return not_evaluated_block("No samples for multi-class evaluation.")
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    per_class: dict[str, Any] = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[o][label] for o in labels if o != label)
        fn = sum(matrix[label][o] for o in labels if o != label)
        support = sum(matrix[label].values())
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = None
        if p is not None and r is not None and (p + r) > 0:
            f1 = 2 * p * r / (p + r)
        per_class[label] = {"precision": p, "recall": r, "f1": f1, "support": support}
        if support > 0:
            precisions.append(p if p is not None else 0.0)
            recalls.append(r if r is not None else 0.0)
            f1s.append(f1 if f1 is not None else 0.0)
            supports.append(support)

    def _avg(vals: list[float]) -> float | None:
        return float(sum(vals) / len(vals)) if vals else None

    total = sum(supports) or 1
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total) if supports else None
    return {
        "metric_status": "EVALUATED",
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro_precision": _avg(precisions),
        "macro_recall": _avg(recalls),
        "macro_f1": _avg(f1s),
        "weighted_f1": weighted_f1,
        "sample_count": int(len(y_true)),
    }


def evidence_strength_precision(
    labels: Sequence[str],
    strengths: Sequence[str | None],
    candidates: Sequence[str | None],
    *,
    positive_label: str = LABEL_INDUSTRIAL,
) -> dict[str, Any]:
    """Descriptive precision by evidence_strength among non-abstaining predictions."""
    buckets: dict[str, Counter] = {}
    for lab, strength, cand in zip(labels, strengths, candidates):
        pred = map_candidate_to_binary(cand, mode="strict")
        if pred is None:
            continue
        key = strength or "UNKNOWN"
        buckets.setdefault(key, Counter())
        if pred == LABEL_INDUSTRIAL:
            if lab == positive_label:
                buckets[key]["tp"] += 1
            else:
                buckets[key]["fp"] += 1
        else:
            buckets[key]["other"] += 1
    out: dict[str, Any] = {"metric_status": "EVALUATED" if buckets else "NOT_EVALUATED", "by_strength": {}}
    for key, ctr in sorted(buckets.items()):
        tp = ctr.get("tp", 0)
        fp = ctr.get("fp", 0)
        out["by_strength"][key] = {
            "predicted_industrial_count": tp + fp,
            "precision": _safe_div(tp, tp + fp),
            "sample_count": int(sum(ctr.values())),
        }
    if not buckets:
        out["reason"] = "No non-abstaining predictions available for strength analysis."
    return out
