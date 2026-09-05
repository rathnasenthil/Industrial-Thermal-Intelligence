"""Synthetic NASA STA fixtures for Stage I.5 unit tests (NOT real NASA data)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon

from src.sta_evidence.config import LAYER_DETECTION, LAYER_MASK


def write_synthetic_sta_mask_geojson(path: Path) -> Path:
    """Write a tiny synthetic STA MASK GeoJSON for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {
            "id": ["MASK_A", "MASK_B"],
            "name": ["Synthetic Plant A", "Synthetic Plant B"],
        },
        geometry=[
            Polygon([(77.0, 28.0), (77.02, 28.0), (77.02, 28.02), (77.0, 28.02)]),
            Polygon([(78.0, 20.0), (78.01, 20.0), (78.01, 20.01), (78.0, 20.01)]),
        ],
        crs="EPSG:4326",
    )
    gdf["_sta_layer_type"] = LAYER_MASK
    # Don't write helper cols to file — loader adds layer type from config.
    gdf = gdf.drop(columns=["_sta_layer_type"], errors="ignore")
    gdf.to_file(path, driver="GeoJSON")
    return path


def write_synthetic_sta_detections_geojson(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {
            "id": ["DET_1", "DET_2"],
            "observation_datetime": ["2023-06-15T10:00:00+00:00", "2023-01-01T00:00:00+00:00"],
        },
        geometry=[
            Point(77.01, 28.01),  # inside MASK_A
            Point(80.0, 15.0),  # far away
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def make_synthetic_events() -> pd.DataFrame:
    """Minimal event table with I.4-like columns and Stage G geometry."""
    rows = [
        {
            "event_id": "EVT_INSIDE",
            "event_start": "2023-06-15T09:00:00+00:00",
            "event_end": "2023-06-15T11:00:00+00:00",
            "centroid_latitude": 28.01,
            "centroid_longitude": 77.01,
            "centroid_wkt": "POINT (77.01 28.01)",
            "footprint_wkt": "POLYGON ((77.005 28.005, 77.015 28.005, 77.015 28.015, 77.005 28.015, 77.005 28.005))",
            "facility_id": None,
            "facility_association_method": "NO_FACILITY_ASSOCIATION",
            "anomaly_score": 1.5,
            "anomaly_status": "NORMAL",
            "anomaly_confidence": "MEDIUM",
            "peak_frp_deviation": 0.5,
            "event_size_deviation": 0.2,
            "duration_deviation": 0.1,
            "distance_deviation": None,
            "persistence_deviation": 0.0,
            "monthly_deviation": None,
            "persistence_label": "SHORT_LIVED",
            "peak_frp": 5.0,
            "detection_count": 3,
            "observed_duration_hours": 2.0,
        },
        {
            "event_id": "EVT_NEAR",
            "event_start": "2023-06-16T09:00:00+00:00",
            "event_end": "2023-06-16T10:00:00+00:00",
            "centroid_latitude": 28.025,
            "centroid_longitude": 77.025,
            "centroid_wkt": "POINT (77.025 28.025)",
            "footprint_wkt": "POINT (77.025 28.025)",
            "facility_id": "F1",
            "facility_association_method": "NEAR_FACILITY",
            "anomaly_score": 2.5,
            "anomaly_status": "ELEVATED",
            "anomaly_confidence": "HIGH",
            "peak_frp_deviation": 2.0,
            "event_size_deviation": 1.0,
            "duration_deviation": 0.5,
            "distance_deviation": 0.3,
            "persistence_deviation": 0.0,
            "monthly_deviation": None,
            "persistence_label": "SHORT_LIVED",
            "peak_frp": 8.0,
            "detection_count": 4,
            "observed_duration_hours": 1.0,
        },
        {
            "event_id": "EVT_NONE",
            "event_start": "2023-07-01T09:00:00+00:00",
            "event_end": "2023-07-01T10:00:00+00:00",
            "centroid_latitude": 10.0,
            "centroid_longitude": 70.0,
            "centroid_wkt": "POINT (70.0 10.0)",
            "footprint_wkt": "POINT (70.0 10.0)",
            "facility_id": "F2",
            "facility_association_method": "WITHIN_FACILITY",
            "anomaly_score": None,
            "anomaly_status": "INSUFFICIENT_HISTORY",
            "anomaly_confidence": "NONE",
            "peak_frp_deviation": None,
            "event_size_deviation": None,
            "duration_deviation": None,
            "distance_deviation": None,
            "persistence_deviation": None,
            "monthly_deviation": None,
            "persistence_label": "INSUFFICIENT_OBSERVATIONS",
            "peak_frp": 2.0,
            "detection_count": 2,
            "observed_duration_hours": 0.5,
        },
        {
            "event_id": "EVT_AMBIG_FAC",
            "event_start": "2023-06-15T09:30:00+00:00",
            "event_end": "2023-06-15T10:30:00+00:00",
            "centroid_latitude": 28.01,
            "centroid_longitude": 77.01,
            "centroid_wkt": "POINT (77.01 28.01)",
            "footprint_wkt": "POINT (77.01 28.01)",
            "facility_id": None,
            "facility_association_method": "AMBIGUOUS",
            "anomaly_score": None,
            "anomaly_status": "INSUFFICIENT_HISTORY",
            "anomaly_confidence": "NONE",
            "peak_frp_deviation": None,
            "event_size_deviation": None,
            "duration_deviation": None,
            "distance_deviation": None,
            "persistence_deviation": None,
            "monthly_deviation": None,
            "persistence_label": "SHORT_LIVED",
            "peak_frp": 3.0,
            "detection_count": 2,
            "observed_duration_hours": 1.0,
        },
    ]
    return pd.DataFrame(rows)


def load_mask_as_gdf(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf["_sta_layer_type"] = LAYER_MASK
    gdf["_sta_source_path"] = str(path)
    return gdf


def load_det_as_gdf(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf["_sta_layer_type"] = LAYER_DETECTION
    gdf["_sta_source_path"] = str(path)
    return gdf
