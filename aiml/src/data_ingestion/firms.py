"""
NASA FIRMS (Fire Information for Resource Management System) ingestion.

This module will eventually retrieve and preprocess thermal/fire hotspot
observations from the NASA FIRMS API (e.g. VIIRS/MODIS active fire products)
for a given area of interest and time range.
"""

from __future__ import annotations

from datetime import date
from typing import Any


def fetch_firms_hotspots(
    area_of_interest: str,
    start_date: date,
    end_date: date,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve raw thermal hotspot records from the NASA FIRMS API.

    TODO:
        - Build the FIRMS API request (area, date range, sensor product).
        - Handle authentication via `FIRMS_API_KEY`.
        - Handle pagination / rate limiting.
        - Return raw records for downstream preprocessing.
    """
    raise NotImplementedError("FIRMS ingestion is not implemented yet.")


def normalize_firms_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize raw FIRMS records into a consistent internal schema
    (coordinates, brightness/FRP, acquisition time, confidence, satellite).

    TODO: Implement field mapping and validation once the raw schema is confirmed.
    """
    raise NotImplementedError("FIRMS record normalization is not implemented yet.")
