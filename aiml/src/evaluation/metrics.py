"""
Model evaluation utilities.

This module will eventually calculate precision, recall, F1, confusion
matrix and other relevant metrics for the thermal event classifier.
"""

from __future__ import annotations

from typing import Any


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """
    Compute standard classification metrics (precision, recall, F1,
    confusion matrix, accuracy) for model evaluation.

    TODO: Implement using scikit-learn's metrics module once the classifier
    produces real predictions.
    """
    raise NotImplementedError("Classification metric computation is not implemented yet.")


def generate_evaluation_report(
    y_true: list[str],
    y_pred: list[str],
) -> str:
    """
    Produce a human-readable evaluation report summarizing model performance.

    TODO: Implement report generation (e.g. classification_report + plots).
    """
    raise NotImplementedError("Evaluation report generation is not implemented yet.")
