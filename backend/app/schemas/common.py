"""Shared pagination and geometry response helpers."""

from __future__ import annotations

from math import ceil
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class PaginationMeta(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


def pagination_totals(total: int, page: int, page_size: int) -> PaginationMeta:
    total_pages = ceil(total / page_size) if page_size > 0 and total > 0 else 0
    return PaginationMeta(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


class GeometryPoint(BaseModel):
    type: str = "Point"
    coordinates: list[float] = Field(
        ...,
        description="GeoJSON [longitude, latitude] in EPSG:4326",
    )


class BBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    model_config = ConfigDict(extra="forbid")


def parse_bbox(value: str) -> BBox:
    """Parse `min_lon,min_lat,max_lon,max_lat`."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError("bbox min values must be <= max values")
    return BBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


def point_from_lon_lat(
    lon: Optional[float],
    lat: Optional[float],
) -> Optional[GeometryPoint]:
    if lon is None or lat is None:
        return None
    return GeometryPoint(coordinates=[lon, lat])
