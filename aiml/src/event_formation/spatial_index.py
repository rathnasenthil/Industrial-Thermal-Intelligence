"""
Haversine-based spatial indexing helpers.

Great-circle (haversine) distance is used instead of naively treating
latitude/longitude degrees as planar (Euclidean) coordinates, because a
degree of longitude does not correspond to a constant physical distance —
it shrinks toward the poles (``cos(latitude)`` scaling). Over the
latitude range of India (~8-35N) that scaling factor varies by roughly
20%, enough to meaningfully distort a naive Euclidean epsilon.

`sklearn.neighbors.BallTree` with ``metric="haversine"`` computes true
great-circle distances directly from (lat, lon) in radians — no map
projection or Euclidean approximation involved — and supports efficient
``O(log n)``-ish radius queries instead of an ``O(n^2)`` pairwise distance
matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree

# Mean Earth radius (km), IUGG value. Used only to convert between
# great-circle angular distance (radians, as used internally by BallTree's
# haversine metric) and physical distance (km).
EARTH_RADIUS_KM = 6371.0088


def build_haversine_tree(latitude: np.ndarray, longitude: np.ndarray) -> BallTree:
    """Build a BallTree over (latitude, longitude) using the haversine metric.

    Args:
        latitude: Latitudes in degrees.
        longitude: Longitudes in degrees.

    Returns:
        A fitted `BallTree`. Distances returned by queries against it are
        great-circle angular distances in radians — convert with
        `radians_to_km`/`km_to_radians`.
    """
    coords_rad = np.radians(np.column_stack([latitude, longitude]))
    return BallTree(coords_rad, metric="haversine")


def to_radians_coords(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Convert (latitude, longitude) in degrees to an (n, 2) radians array."""
    return np.radians(np.column_stack([latitude, longitude]))


def km_to_radians(distance_km: float) -> float:
    """Convert a great-circle distance in km to radians for BallTree queries."""
    return distance_km / EARTH_RADIUS_KM


def radians_to_km(distance_radians: float) -> float:
    """Convert a great-circle angular distance in radians to km."""
    return distance_radians * EARTH_RADIUS_KM
