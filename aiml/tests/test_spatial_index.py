"""Tests for src.event_formation.spatial_index (haversine BallTree helpers)."""

from __future__ import annotations

import numpy as np

from src.event_formation.spatial_index import (
    build_haversine_tree,
    km_to_radians,
    radians_to_km,
)


def test_km_radians_round_trip() -> None:
    km = 1.5
    assert abs(radians_to_km(km_to_radians(km)) - km) < 1e-9


def test_haversine_tree_finds_nearby_points_within_radius() -> None:
    # Two points ~0 km apart (identical), one ~far away.
    lat = np.array([28.6139, 28.6140, 10.0])
    lon = np.array([77.2090, 77.2091, 78.0])

    tree = build_haversine_tree(lat, lon)
    radius = km_to_radians(1.0)
    neighbors = tree.query_radius(np.radians(np.column_stack([lat, lon])), r=radius)

    assert set(neighbors[0]) == {0, 1}
    assert set(neighbors[1]) == {0, 1}
    assert set(neighbors[2]) == {2}


def test_haversine_distance_matches_known_great_circle_distance() -> None:
    # Delhi to Mumbai is approximately 1150-1160 km great-circle.
    delhi = (28.7041, 77.1025)
    mumbai = (19.0760, 72.8777)

    tree = build_haversine_tree(np.array([delhi[0]]), np.array([delhi[1]]))
    dist_rad, _ = tree.query(np.radians([[mumbai[0], mumbai[1]]]), k=1)
    dist_km = radians_to_km(dist_rad[0][0])

    assert 1100 < dist_km < 1200
