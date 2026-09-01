"""
Preprocessing pipeline for raw thermal observations and contextual data.

This module will eventually clean, deduplicate and merge thermal
observations (from FIRMS) with contextual data (from OSM and satellite/
land-cover sources) into a unified, analysis-ready dataset, and group
repeated detections into discrete thermal "events".
"""

from __future__ import annotations

from typing import Any


def clean_thermal_observations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Clean raw thermal observation records: drop invalid coordinates,
    deduplicate, standardize timestamps/units.

    TODO: Implement validation and cleaning rules.
    """
    raise NotImplementedError("Thermal observation cleaning is not implemented yet.")


def form_thermal_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group spatially/temporally clustered thermal observations into discrete
    "events" (e.g. a persistent thermal source detected across multiple
    satellite passes).

    TODO: Implement spatial/temporal clustering logic.
    """
    raise NotImplementedError("Thermal event formation is not implemented yet.")
