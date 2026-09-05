"""
NASA STA source loader for GIFT Stage I.5.

SOURCE POLICY
------------------------------------------------------------------------
NASA FIRMS documents Static Thermal Anomalies (STA) — Mask and STA —
Detections as experimental/provisional layers in the FIRMS Fire Map
(Advanced Mode). A stable public bulk-download CSV endpoint for these
layers is not assumed or hard-coded here.

This loader accepts **locally supplied** vector files only:

* GeoJSON (``.geojson`` / ``.json``)
* GeoPackage (``.gpkg``)
* Shapefile (``.shp``)
* CSV with ``latitude``/``longitude`` or ``geometry_wkt`` / ``wkt``

Place downloaded NASA STA extracts under ``aiml/data/raw/`` (gitignored)
and point ``STAConfig.mask_path`` / ``detection_path`` at them.

Fabricating STA geometries is forbidden. Missing files raise a clear
``FileNotFoundError`` with documentation URLs from the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import shape

from src.sta_evidence.config import LAYER_DETECTION, LAYER_MASK, STAConfig

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".geojson", ".json", ".gpkg", ".shp", ".csv"})


class STASourceMissingError(FileNotFoundError):
    """Raised when no configured local NASA STA source file is present."""


def resolve_existing_paths(config: STAConfig) -> list[tuple[Path, str]]:
    """Return ``(path, layer_type)`` pairs for configured files that exist."""
    pairs: list[tuple[Path, str]] = []
    if config.mask_path is not None and Path(config.mask_path).exists():
        pairs.append((Path(config.mask_path), LAYER_MASK))
    if config.detection_path is not None and Path(config.detection_path).exists():
        pairs.append((Path(config.detection_path), LAYER_DETECTION))
    return pairs


def require_sta_sources(config: STAConfig) -> list[tuple[Path, str]]:
    """Resolve STA sources or raise ``STASourceMissingError`` with guidance."""
    pairs = resolve_existing_paths(config)
    if pairs:
        return pairs
    mask = config.mask_path
    det = config.detection_path
    raise STASourceMissingError(
        "No local NASA FIRMS STA source file found.\n"
        f"  Expected MASK path: {mask}\n"
        f"  Expected DETECTION path: {det}\n"
        "Place a downloaded NASA STA Mask and/or STA Detections extract "
        "(GeoJSON / GPKG / Shapefile / CSV) under aiml/data/raw/ and re-run.\n"
        "Do not fabricate STA geometries.\n"
        f"Documentation: {config.sta_documentation_url}\n"
        f"Earthdata overview: {config.sta_source_url}"
    )


def load_sta_vector(path: str | Path, layer_type: str) -> gpd.GeoDataFrame:
    """Load one STA vector file into EPSG:4326 GeoDataFrame (geometry column).

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: Unsupported format or unreadable geometry.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STA source file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported STA source format '{suffix}' for {path}. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".csv":
        gdf = _load_sta_csv(path)
    else:
        gdf = gpd.read_file(path)

    if gdf.empty:
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() not in ("EPSG:4326", "EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")

    if "geometry" not in gdf.columns and gdf.geometry.name != "geometry":
        gdf = gdf.set_geometry(gdf.geometry.name)

    gdf = gdf.copy()
    gdf["_sta_layer_type"] = layer_type
    gdf["_sta_source_path"] = str(path)
    return gdf


def _load_sta_csv(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "geometry_wkt" in df.columns or "wkt" in df.columns:
        col = "geometry_wkt" if "geometry_wkt" in df.columns else "wkt"
        geometries = []
        for value in df[col]:
            if pd.isna(value) or str(value).strip() == "" or str(value).lower() == "nan":
                geometries.append(None)
            else:
                geometries.append(shapely_wkt.loads(str(value)))
        return gpd.GeoDataFrame(df.drop(columns=[col]), geometry=geometries, crs="EPSG:4326")

    lat_col = next((c for c in ("latitude", "lat", "LATITUDE", "Lat") if c in df.columns), None)
    lon_col = next((c for c in ("longitude", "lon", "LONGITUDE", "Lon", "lng") if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"CSV STA source {path} must contain geometry_wkt/wkt or latitude+longitude columns."
        )
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )


def load_all_sta_layers(config: STAConfig) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Load all configured existing STA layers and return combined GDF + load stats."""
    pairs = require_sta_sources(config)
    frames: list[gpd.GeoDataFrame] = []
    per_file: list[dict[str, Any]] = []
    for path, layer_type in pairs:
        gdf = load_sta_vector(path, layer_type)
        frames.append(gdf)
        per_file.append(
            {
                "path": str(path),
                "layer_type": layer_type,
                "records_read": int(len(gdf)),
                "file_size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
        )
    combined = pd.concat(frames, ignore_index=True) if frames else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if not isinstance(combined, gpd.GeoDataFrame):
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    stats = {
        "files_loaded": per_file,
        "total_records_read": int(len(combined)),
        "layer_types_present": sorted({p[1] for p in pairs}),
    }
    return combined, stats
