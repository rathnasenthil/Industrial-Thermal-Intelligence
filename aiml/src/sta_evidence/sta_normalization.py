"""
Normalize loaded NASA STA vectors into the canonical Stage I.5 schema.

Validation rejects empty/invalid/unsupported geometries with explicit reasons.
Records are never silently dropped without appearing in rejection stats.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from src.sta_evidence.config import STAConfig
from src.sta_evidence.sta_schema import (
    CANONICAL_COLUMNS,
    VALID_MATCH_GEOMETRY_TYPES,
    deterministic_sta_id,
    empty_canonical_frame,
    serialize_raw_attributes,
)

NATIVE_ID_CANDIDATES: tuple[str, ...] = (
    "sta_id",
    "STA_ID",
    "id",
    "ID",
    "OBJECTID",
    "FID",
    "gid",
    "feature_id",
)

DATETIME_CANDIDATES: tuple[str, ...] = (
    "observation_datetime",
    "acq_datetime",
    "datetime",
    "timestamp",
    "acq_date",
    "date",
    "DATE",
)

SOURCE_DATE_CANDIDATES: tuple[str, ...] = ("source_date", "layer_date", "pub_date", "version_date")


def normalize_sta_geodataframe(gdf: gpd.GeoDataFrame, config: STAConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a loaded STA GeoDataFrame into the canonical tabular schema.

    Returns:
        (canonical_df, validation_stats) where ``canonical_df`` includes both
        valid and rejected rows (``is_valid`` flag). Matching uses only
        ``is_valid == True`` rows.
    """
    if gdf is None or len(gdf) == 0:
        stats = {
            "records_read": 0,
            "records_valid": 0,
            "records_rejected": 0,
            "rejection_reasons": {},
            "geometry_type_counts": {},
            "duplicate_sta_id_count": 0,
            "layer_counts": {},
        }
        return empty_canonical_frame(), stats

    rows: list[dict[str, Any]] = []
    rejection_reasons: dict[str, int] = {}

    for index, series in gdf.iterrows():
        geom = series.geometry if hasattr(series, "geometry") else None
        layer_type = str(series.get("_sta_layer_type", "UNKNOWN"))
        native_id = _first_present(series, NATIVE_ID_CANDIDATES)
        obs_dt = _parse_optional_datetime(_first_present(series, DATETIME_CANDIDATES))
        source_date = _first_present(series, SOURCE_DATE_CANDIDATES)

        reason = _validate_geometry(geom)
        is_valid = reason is None
        if not is_valid and reason:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        geom_type = geom.geom_type if isinstance(geom, BaseGeometry) and geom is not None and not geom.is_empty else None
        latitude = None
        longitude = None
        geometry_wkt = None
        if isinstance(geom, BaseGeometry) and geom is not None and not geom.is_empty:
            geometry_wkt = geom.wkt
            rep = geom.representative_point()
            latitude = float(rep.y)
            longitude = float(rep.x)
            # Prefer explicit lat/lon columns when present and valid.
            for lat_key, lon_key in (("latitude", "longitude"), ("lat", "lon")):
                if lat_key in series.index and lon_key in series.index:
                    try:
                        la = float(series[lat_key])
                        lo = float(series[lon_key])
                        if -90 <= la <= 90 and -180 <= lo <= 180:
                            latitude, longitude = la, lo
                    except (TypeError, ValueError):
                        pass

        if latitude is not None and not (-90 <= latitude <= 90):
            is_valid = False
            reason = "invalid_latitude"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        if longitude is not None and not (-180 <= longitude <= 180):
            is_valid = False
            reason = "invalid_longitude"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        raw_attrs = {
            k: series[k]
            for k in series.index
            if not str(k).startswith("_") and k not in ("geometry",)
        }

        sta_id = deterministic_sta_id(layer_type, None if pd.isna(native_id) else str(native_id), geometry_wkt, int(index) if isinstance(index, (int, np.integer)) else len(rows))

        rows.append(
            {
                "sta_id": sta_id,
                "sta_layer_type": layer_type,
                "geometry_type": geom_type.upper() if geom_type else None,
                "latitude": latitude,
                "longitude": longitude,
                "geometry_wkt": geometry_wkt,
                "sta_source": config.sta_source,
                "sta_source_version": config.sta_source_version,
                "sta_source_url": config.sta_source_url,
                "sta_download_date": config.sta_download_date,
                "sta_layer": f"STA_{layer_type}",
                "source_date": None if source_date is None or (isinstance(source_date, float) and pd.isna(source_date)) else str(source_date),
                "observation_datetime": obs_dt,
                "raw_attributes": serialize_raw_attributes(raw_attrs),
                "is_valid": is_valid,
                "rejection_reason": reason,
            }
        )

    out = pd.DataFrame(rows)
    # Deterministic duplicate handling: keep first occurrence of sta_id among valid rows;
    # mark later duplicates rejected (never silently delete).
    duplicate_count = 0
    if not out.empty:
        seen: set[str] = set()
        for i, row in out.iterrows():
            sid = row["sta_id"]
            if sid in seen:
                duplicate_count += 1
                if row["is_valid"]:
                    out.at[i, "is_valid"] = False
                    out.at[i, "rejection_reason"] = "duplicate_sta_id"
                    rejection_reasons["duplicate_sta_id"] = rejection_reasons.get("duplicate_sta_id", 0) + 1
            else:
                seen.add(sid)

    out = out.sort_values("sta_id", kind="mergesort").reset_index(drop=True)
    # Ensure column order
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[list(CANONICAL_COLUMNS)]

    valid = out["is_valid"] == True  # noqa: E712
    geom_counts = out.loc[valid, "geometry_type"].value_counts().to_dict() if valid.any() else {}
    layer_counts = out.loc[valid, "sta_layer_type"].value_counts().to_dict() if valid.any() else {}

    stats = {
        "records_read": int(len(out)),
        "records_valid": int(valid.sum()),
        "records_rejected": int((~valid).sum()),
        "rejection_reasons": {k: int(v) for k, v in sorted(rejection_reasons.items())},
        "geometry_type_counts": {str(k): int(v) for k, v in geom_counts.items()},
        "duplicate_sta_id_count": int(duplicate_count),
        "layer_counts": {str(k): int(v) for k, v in layer_counts.items()},
    }
    return out, stats


def canonical_to_geodataframe(canonical_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build an EPSG:4326 GeoDataFrame from valid canonical rows only."""
    valid = canonical_df.loc[canonical_df["is_valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return gpd.GeoDataFrame(columns=list(canonical_df.columns) + ["geometry"], geometry="geometry", crs="EPSG:4326")

    geometries = []
    for value in valid["geometry_wkt"]:
        if pd.isna(value) or value is None:
            geometries.append(None)
        else:
            from shapely import wkt as shapely_wkt

            geometries.append(shapely_wkt.loads(str(value)))
    return gpd.GeoDataFrame(valid, geometry=geometries, crs="EPSG:4326")


def _validate_geometry(geom: BaseGeometry | None) -> str | None:
    if geom is None or (isinstance(geom, float) and np.isnan(geom)):
        return "empty_or_missing_geometry"
    if not isinstance(geom, BaseGeometry):
        return "unsupported_geometry_object"
    if geom.is_empty:
        return "empty_or_missing_geometry"
    if not geom.is_valid:
        return "invalid_geometry"
    if geom.geom_type not in VALID_MATCH_GEOMETRY_TYPES:
        return f"unsupported_geometry_type:{geom.geom_type}"
    return None


def _first_present(series: pd.Series, candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in series.index:
            value = series[key]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text == "" or text.lower() == "nan":
                continue
            return value
    return None


def _parse_optional_datetime(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.isoformat()
