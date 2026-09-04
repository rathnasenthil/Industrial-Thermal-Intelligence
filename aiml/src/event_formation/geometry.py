"""
Event geometry construction for later PostGIS integration.

Each thermal event's spatial footprint is represented as the convex hull
of its member detections' (longitude, latitude) points, encoded as
Well-Known Text (WKT) in WGS84 (EPSG:4326) — the same coordinate
reference system as the raw FIRMS lat/lon — so it can be loaded directly
into a PostGIS ``geometry(Geometry, 4326)`` column later.

Why convex hull (and not a buffered-point-collection or bounding box):

* A convex hull is a standard, well-understood, dependency-light
  representation (this project already depends on ``shapely`` via
  ``geopandas``/``requirements.txt`` — no new dependency needed) that
  captures the actual spatial spread and orientation of the detections
  far better than a single centroid point or an axis-aligned bounding
  box.
* It requires no extra assumptions (e.g. a buffer radius) beyond the
  detections' own coordinates, so it doesn't imply a level of precision
  about the sensor footprint that the raw data doesn't support.

IMPORTANT CAVEAT: this convex hull is the envelope of confirmed satellite
*pixel-center* detections. It is **not** the true physical fire/thermal
source perimeter:

* It does not account for individual pixel footprint size (~0.3-0.8 km
  per side for these VIIRS I-band detections — see
  `src.event_formation.config`), so it will typically underestimate the
  true extent by roughly half a pixel width on every side.
* It does not account for activity between satellite overpasses (FIRMS
  detections are discrete snapshots, not continuous coverage).
* It does not account for parts of the thermal source that fell below
  the sensor's detection threshold.

Downstream stages must treat `footprint_wkt` as an approximate, detection-
derived envelope, not a fire-boundary product.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from shapely.geometry import MultiPoint


class EventGeometry(NamedTuple):
    """Geometric summary of one thermal event's member detections.

    Attributes:
        centroid_latitude: Arithmetic mean latitude of member detections.
            This is a simple coordinate average, not a true spherical
            centroid — adequate at the sub-100 km event scales expected
            here, but not appropriate for very large or antimeridian-
            crossing spatial extents (not a concern for this India-only
            dataset).
        centroid_longitude: Arithmetic mean longitude of member
            detections (same caveat as above).
        centroid_wkt: WKT ``POINT`` built from the centroid lat/lon.
        footprint_wkt: WKT geometry (``POINT``, ``LINESTRING`` or
            ``POLYGON`` depending on detection count/collinearity)
            representing the convex hull of member detection coordinates.
            See module docstring for what this does/doesn't represent.
        min_latitude: Minimum latitude among member detections.
        max_latitude: Maximum latitude among member detections.
        min_longitude: Minimum longitude among member detections.
        max_longitude: Maximum longitude among member detections.
    """

    centroid_latitude: float
    centroid_longitude: float
    centroid_wkt: str
    footprint_wkt: str
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float


def compute_event_geometry(latitudes: np.ndarray, longitudes: np.ndarray) -> EventGeometry:
    """Compute the centroid, bounding box and convex-hull footprint of an event.

    Args:
        latitudes: Latitudes (degrees) of every detection in the event.
        longitudes: Longitudes (degrees) of every detection in the event.

    Returns:
        An `EventGeometry`.

    Raises:
        ValueError: If ``latitudes``/``longitudes`` are empty.
    """
    if len(latitudes) == 0:
        raise ValueError("Cannot compute event geometry for zero detections.")

    centroid_lat = float(np.mean(latitudes))
    centroid_lon = float(np.mean(longitudes))

    # shapely uses (x, y) = (longitude, latitude) ordering.
    points = MultiPoint(list(zip(longitudes, latitudes)))
    if len(latitudes) == 1:
        footprint = points.geoms[0]
    else:
        footprint = points.convex_hull

    centroid_wkt = f"POINT ({centroid_lon} {centroid_lat})"

    return EventGeometry(
        centroid_latitude=centroid_lat,
        centroid_longitude=centroid_lon,
        centroid_wkt=centroid_wkt,
        footprint_wkt=footprint.wkt,
        min_latitude=float(np.min(latitudes)),
        max_latitude=float(np.max(latitudes)),
        min_longitude=float(np.min(longitudes)),
        max_longitude=float(np.max(longitudes)),
    )
