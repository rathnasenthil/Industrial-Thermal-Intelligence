"""
Thermal event classifier.

This module will eventually train and expose a model that classifies
thermal events (e.g. industrial fire, wildfire, agricultural burning,
gas flare, persistent thermal source) from engineered features.

No model architecture is finalized yet. Candidate approaches include
Random Forest and XGBoost (see `docs/ml/ml-plan.md`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ThermalEventClassifier:
    """Placeholder interface for the future thermal event classifier."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path
        self.model: Any = None

    def train(self, features: list[dict[str, float]], labels: list[str]) -> None:
        """
        Train the classifier on labeled thermal event features.

        TODO: Implement training pipeline (train/test split, model selection,
        hyperparameter tuning) once labeled data is available.
        """
        raise NotImplementedError("Classifier training is not implemented yet.")

    def predict(self, features: list[dict[str, float]]) -> list[str]:
        """
        Predict thermal event classes for the given feature vectors.

        TODO: Implement inference using the trained model.
        """
        raise NotImplementedError("Classifier prediction is not implemented yet.")

    def save(self, path: Path) -> None:
        """TODO: Persist the trained model (e.g. via joblib)."""
        raise NotImplementedError("Model persistence is not implemented yet.")

    def load(self, path: Path) -> None:
        """TODO: Load a previously trained model."""
        raise NotImplementedError("Model loading is not implemented yet.")
