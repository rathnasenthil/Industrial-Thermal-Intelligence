"""
Inference entrypoint.

This module will eventually expose prediction functionality to the backend
(FastAPI service), providing a clean interface so the backend does not need
to know about model internals, feature engineering, or training code.
"""

from __future__ import annotations

from typing import Any


def predict_thermal_event(raw_event: dict[str, Any]) -> dict[str, Any]:
    """
    Given a raw thermal event (post event-formation, pre feature-engineering),
    run the full inference pipeline and return a classification result.

    TODO:
        - Run feature engineering (`feature_engineering.features`).
        - Load the trained classifier (`models.classifier`).
        - Run prediction and return a structured result (class, confidence).
        - This function is the intended integration point for the backend
          (`backend/app/services/`) to call into the AIML pipeline.
    """
    raise NotImplementedError("Inference pipeline is not implemented yet.")
