"""Tests for src.event_formation.geometry (centroid, bounding box, footprint)."""

from __future__ import annotations

import numpy as np
import pytest

from src.event_formation.geometry import compute_event_geometry


def test_centroid_is_arithmetic_mean_of_points() -> None:
    lat = np.array([10.0, 20.0, 30.0])
    lon = np.array([70.0, 80.0, 90.0])

    geom = compute_event_geometry(lat, lon)

    assert geom.centroid_latitude == pytest.approx(20.0)
    assert geom.centroid_longitude == pytest.approx(80.0)
    assert geom.centroid_wkt == "POINT (80.0 20.0)"


def test_bounding_box_matches_min_max() -> None:
    lat = np.array([10.0, 25.0, 15.0])
    lon = np.array([70.0, 72.0, 68.0])

    geom = compute_event_geometry(lat, lon)

    assert geom.min_latitude == 10.0
    assert geom.max_latitude == 25.0
    assert geom.min_longitude == 68.0
    assert geom.max_longitude == 72.0


def test_single_point_footprint_is_a_point() -> None:
    geom = compute_event_geometry(np.array([10.0]), np.array([70.0]))
    assert geom.footprint_wkt.startswith("POINT")


def test_two_point_footprint_is_a_linestring() -> None:
    geom = compute_event_geometry(np.array([10.0, 10.1]), np.array([70.0, 70.1]))
    assert geom.footprint_wkt.startswith("LINESTRING")


def test_three_non_collinear_points_footprint_is_a_polygon() -> None:
    geom = compute_event_geometry(np.array([10.0, 10.1, 10.0]), np.array([70.0, 70.0, 70.1]))
    assert geom.footprint_wkt.startswith("POLYGON")


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        compute_event_geometry(np.array([]), np.array([]))
