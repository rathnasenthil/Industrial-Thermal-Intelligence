"""
Feature engineering for thermal event classification.

This module will eventually calculate spatial, temporal and thermal
features used to classify thermal events (industrial fire, wildfire,
agricultural burning, gas flare, persistent thermal source, etc.).
"""

from __future__ import annotations

from typing import Any


def compute_spatial_features(event: dict[str, Any]) -> dict[str, float]:
    """
    Compute spatial features for a thermal event (e.g. distance to nearest
    industrial site/road/settlement, land-cover category).

    TODO: Implement using OSM + land-cover context.
    """
    raise NotImplementedError("Spatial feature computation is not implemented yet.")


def compute_temporal_features(event: dict[str, Any]) -> dict[str, float]:
    """
    Compute temporal features for a thermal event (e.g. detection frequency,
    time-of-day pattern, persistence duration, seasonality).

    TODO: Implement using historical thermal observation history.
    """
    raise NotImplementedError("Temporal feature computation is not implemented yet.")


def compute_thermal_features(event: dict[str, Any]) -> dict[str, float]:
    """
    Compute thermal-signature features for a thermal event (e.g. brightness
    temperature, fire radiative power statistics, confidence trends).

    TODO: Implement using FIRMS radiometric fields.
    """
    raise NotImplementedError("Thermal feature computation is not implemented yet.")


def build_feature_vector(event: dict[str, Any]) -> dict[str, float]:
    """
    Combine spatial, temporal and thermal features into the final feature
    vector consumed by the classifier.

    TODO: Compose the above feature functions once each is implemented.
    """
    raise NotImplementedError("Feature vector assembly is not implemented yet.")
