"""
OpenStreetMap (OSM) ingestion.

This module will eventually retrieve industrial and geographic context
(e.g. factories, refineries, mining sites, gas infrastructure, land use)
around detected thermal hotspots to help distinguish industrial fires from
wildfires, agricultural burning and other persistent thermal sources.
"""

from __future__ import annotations

from typing import Any


def fetch_osm_features(
    latitude: float,
    longitude: float,
    radius_meters: float = 1000.0,
) -> list[dict[str, Any]]:
    """
    Retrieve OSM features (e.g. via the Overpass API) within a radius of a
    given point.

    TODO:
        - Query Overpass API for relevant tags (industrial, landuse, power,
          man_made, etc.).
        - Handle request throttling and retries.
        - Return structured feature records for feature engineering.
    """
    raise NotImplementedError("OSM ingestion is not implemented yet.")


def classify_nearby_infrastructure(features: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize nearby OSM infrastructure into features usable by the
    classification model (e.g. distance to nearest industrial site,
    land-use category).

    TODO: Implement once the feature set is defined in `feature_engineering`.
    """
    raise NotImplementedError("OSM infrastructure classification is not implemented yet.")
