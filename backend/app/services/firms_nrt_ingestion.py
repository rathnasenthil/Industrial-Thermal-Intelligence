"""
NASA FIRMS near-real-time (NRT) Area API client.

Phase 1 only: HTTP fetch → CSV parse → normalize columns for AIML FIRMS
preprocessing compatibility. No database writes, scheduling, or Stage G.

API reference: https://firms.modaps.eosdis.nasa.gov/api/area/

URL pattern (MAP_KEY is a path segment — never log the unredacted URL):
  {base}/{MAP_KEY}/{SOURCE}/{AREA_COORDINATES}/{DAY_RANGE}
  optional: .../{DAY_RANGE}/{DATE}

AREA_COORDINATES: west,south,east,north (or "world")
DAY_RANGE: 1..5
SOURCE example: VIIRS_NOAA20_NRT
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import pandas as pd

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Match aiml/src/data_ingestion/firms_csv.py REQUIRED_COLUMNS so downstream
# preprocessing (src.preprocessing) can consume this frame without schema drift.
REQUIRED_FIRMS_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
    "type",
)

SUPPORTED_DAY_RANGE = range(1, 6)

# Common VIIRS NRT products (not exhaustive; config may override).
DEFAULT_VIIRS_NRT_PRODUCT = "VIIRS_NOAA20_NRT"


class FirmsNRTError(Exception):
    """Base error for FIRMS NRT ingestion failures."""


class FirmsNRTConfigError(FirmsNRTError):
    """Invalid or missing configuration (key, bbox, product, day_range)."""


class FirmsNRTHttpError(FirmsNRTError):
    """HTTP / network failure talking to the FIRMS Area API."""


class FirmsNRTParseError(FirmsNRTError):
    """CSV body could not be parsed or failed schema checks."""


@dataclass(frozen=True)
class FirmsNRTRequest:
    """Parameters for one FIRMS Area API request (excluding the secret MAP_KEY)."""

    product: str
    bbox: str
    day_range: int
    date: Optional[str] = None  # optional YYYY-MM-DD for historical window


def _require_map_key(map_key: Optional[str]) -> str:
    key = (map_key or "").strip()
    if not key:
        raise FirmsNRTConfigError(
            "FIRMS_MAP_KEY is missing or empty. Set it in backend/.env "
            "(see backend/.env.example). Obtain a free key from "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )
    return key


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    """
    Parse FIRMS AREA_COORDINATES as west,south,east,north.

    Raises:
        FirmsNRTConfigError: If the bbox string is invalid.
    """
    text = (bbox or "").strip()
    if not text:
        raise FirmsNRTConfigError(
            "FIRMS bbox is empty. Expected west,south,east,north "
            "(e.g. 68.0,6.0,98.0,37.5) or 'world'."
        )
    if text.lower() == "world":
        return (-180.0, -90.0, 180.0, 90.0)

    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise FirmsNRTConfigError(
            f"Invalid FIRMS bbox '{text}'. Expected four comma-separated "
            "numbers: west,south,east,north (decimal degrees), or 'world'."
        )
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError as exc:
        raise FirmsNRTConfigError(
            f"Invalid FIRMS bbox '{text}': values must be numeric "
            "(west,south,east,north)."
        ) from exc

    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise FirmsNRTConfigError(
            f"Invalid FIRMS bbox longitude range in '{text}' "
            "(west/east must be within [-180, 180])."
        )
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise FirmsNRTConfigError(
            f"Invalid FIRMS bbox latitude range in '{text}' "
            "(south/north must be within [-90, 90])."
        )
    if west > east:
        raise FirmsNRTConfigError(
            f"Invalid FIRMS bbox '{text}': west ({west}) must be <= east ({east})."
        )
    if south > north:
        raise FirmsNRTConfigError(
            f"Invalid FIRMS bbox '{text}': south ({south}) must be <= north ({north})."
        )
    return west, south, east, north


def validate_day_range(day_range: int) -> int:
    if day_range not in SUPPORTED_DAY_RANGE:
        raise FirmsNRTConfigError(
            f"Invalid FIRMS day_range={day_range}. "
            "The Area API accepts an integer from 1 to 5 inclusive."
        )
    return day_range


def validate_product(product: str) -> str:
    value = (product or "").strip()
    if not value:
        raise FirmsNRTConfigError(
            "FIRMS product/source is empty. Example NRT products: "
            "VIIRS_NOAA20_NRT, VIIRS_SNPP_NRT, VIIRS_NOAA21_NRT."
        )
    if "/" in value or " " in value:
        raise FirmsNRTConfigError(
            f"Invalid FIRMS product '{value}': must be a single SOURCE token "
            "(e.g. VIIRS_NOAA20_NRT), not a path fragment."
        )
    return value


def redact_secrets(text: str, map_key: str) -> str:
    """Remove MAP_KEY material from URLs/messages before logging or raising."""
    if not text:
        return text
    redacted = text
    key = (map_key or "").strip()
    if key:
        redacted = redacted.replace(key, "***")
    # Defensive: redact path segment after /csv/ if it looks like a key.
    redacted = re.sub(
        r"(/api/area/csv/)[^/]+",
        r"\1***",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def build_firms_area_url(
    *,
    map_key: str,
    product: str,
    bbox: str,
    day_range: int,
    date: Optional[str] = None,
    base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
) -> str:
    """
    Build the FIRMS Area CSV URL.

    Does not validate the secret; callers should use `_require_map_key` first.
    """
    key = _require_map_key(map_key)
    source = validate_product(product)
    area = bbox.strip()
    if area.lower() != "world":
        parse_bbox(area)  # validate
    days = validate_day_range(day_range)
    root = base_url.rstrip("/")
    url = f"{root}/{key}/{source}/{area}/{days}"
    if date:
        date_text = date.strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
            raise FirmsNRTConfigError(
                f"Invalid FIRMS date '{date}'. Expected YYYY-MM-DD when provided."
            )
        url = f"{url}/{date_text}"
    return url


def _looks_like_csv(text: str) -> bool:
    first = text.lstrip()[:500].lower()
    if not first:
        return False
    if first.startswith("<!doctype") or first.startswith("<html"):
        return False
    return "latitude" in first.splitlines()[0] if first.splitlines() else False


def parse_firms_csv_text(text: str) -> pd.DataFrame:
    """
    Parse FIRMS CSV response body into a DataFrame (all columns as strings).

    Empty body or header-only CSV yields an empty DataFrame.
    """
    if text is None:
        raise FirmsNRTParseError("FIRMS response body is None.")

    body = text.strip()
    if not body:
        raise FirmsNRTParseError(
            "FIRMS response body is empty. Check MAP_KEY, product, bbox, "
            "and day_range; the Area API should at least return a CSV header."
        )

    if not _looks_like_csv(body):
        preview = redact_secrets(body[:200].replace("\n", " "), "")
        raise FirmsNRTParseError(
            "FIRMS response does not look like a CSV hotspot table "
            f"(expected a 'latitude' header). Preview: {preview!r}"
        )

    try:
        df = pd.read_csv(io.StringIO(body), dtype=str, keep_default_na=True)
    except Exception as exc:  # noqa: BLE001 - surface as parse error
        raise FirmsNRTParseError(f"Failed to parse FIRMS CSV body: {exc}") from exc

    return df


def normalize_firms_nrt_dataframe(
    df: pd.DataFrame,
    *,
    product: str,
    source_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize raw FIRMS Area CSV columns for AIML preprocessing compatibility.

    - Strip / lower-case column names where needed (FIRMS already uses lowercase).
    - Ensure required VIIRS columns exist (add empty `type` only if absent —
      some NRT extracts omit it; never invent an observation ID).
    - Keep all original columns; cast values to string dtype like
      ``aiml.src.data_ingestion.firms_csv.load_firms_csv``.
    - Add ``source_file`` provenance tag (not a filesystem path).
    """
    if df is None:
        raise FirmsNRTParseError("Cannot normalize a null DataFrame.")

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # Preserve unexpected extra columns; only require the AIML VIIRS set.
    missing = [c for c in REQUIRED_FIRMS_COLUMNS if c not in out.columns]
    # `type` is present on archive exports; NRT sometimes omits it — pad empty.
    if "type" in missing:
        out["type"] = pd.NA
        missing = [c for c in missing if c != "type"]
    if missing:
        raise FirmsNRTParseError(
            f"FIRMS CSV missing required column(s): {missing}. "
            f"Found columns: {list(out.columns)}. "
            "Expected the VIIRS hotspot schema used by "
            "aiml/src/data_ingestion/firms_csv.py."
        )

    # String dtype for all cells — numeric/date coercion belongs in preprocessing.
    for column in out.columns:
        out[column] = out[column].astype("string")

    label = source_label or f"firms_nrt_{validate_product(product)}"
    out["source_file"] = label
    return out.reset_index(drop=True)


def fetch_firms_nrt_csv(
    *,
    map_key: str,
    request: FirmsNRTRequest,
    base_url: str,
    timeout_seconds: float,
    client: Optional[httpx.Client] = None,
) -> str:
    """
    Execute the FIRMS Area CSV GET and return the response text.

    Raises FirmsNRTHttpError on network/HTTP failures (URL redacted).
    """
    url = build_firms_area_url(
        map_key=map_key,
        product=request.product,
        bbox=request.bbox,
        day_range=request.day_range,
        date=request.date,
        base_url=base_url,
    )
    safe_url = redact_secrets(url, map_key)
    logger.info(
        "Fetching FIRMS NRT CSV product=%s bbox=%s day_range=%s",
        request.product,
        request.bbox if request.bbox.lower() != "world" else "world",
        request.day_range,
    )

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_seconds)
    try:
        try:
            response = http.get(url)
        except httpx.TimeoutException as exc:
            raise FirmsNRTHttpError(
                f"FIRMS Area API request timed out after {timeout_seconds}s "
                f"(url={safe_url}). Try increasing FIRMS_TIMEOUT_SECONDS or "
                "narrowing the bbox."
            ) from exc
        except httpx.HTTPError as exc:
            raise FirmsNRTHttpError(
                f"FIRMS Area API network error: {exc.__class__.__name__} "
                f"(url={safe_url})."
            ) from exc

        if response.status_code >= 400:
            body_preview = redact_secrets((response.text or "")[:180], map_key)
            raise FirmsNRTHttpError(
                f"FIRMS Area API HTTP {response.status_code} for {safe_url}. "
                f"Body preview: {body_preview!r}"
            )
        return response.text
    finally:
        if owns_client:
            http.close()


def fetch_firms_nrt_observations(
    *,
    map_key: Optional[str] = None,
    product: Optional[str] = None,
    bbox: Optional[str] = None,
    day_range: Optional[int] = None,
    date: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    base_url: Optional[str] = None,
    settings: Optional[Settings] = None,
    client: Optional[httpx.Client] = None,
) -> pd.DataFrame:
    """
    Fetch and normalize FIRMS NRT observations into an AIML-compatible DataFrame.

    Returns an empty DataFrame with the required schema when the API returns
    a valid header-only CSV (zero hotspots). Does not write to the database.
    """
    cfg = settings or get_settings()
    key = _require_map_key(map_key if map_key is not None else cfg.firms_map_key)
    req = FirmsNRTRequest(
        product=validate_product(product or cfg.firms_product),
        bbox=(bbox if bbox is not None else cfg.firms_bbox).strip(),
        day_range=validate_day_range(
            day_range if day_range is not None else cfg.firms_day_range
        ),
        date=date,
    )
    # Validate bbox early (allows "world").
    if req.bbox.lower() != "world":
        parse_bbox(req.bbox)

    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(cfg.firms_timeout_seconds)
    )
    api_base = base_url or cfg.firms_base_url

    raw_text = fetch_firms_nrt_csv(
        map_key=key,
        request=req,
        base_url=api_base,
        timeout_seconds=timeout,
        client=client,
    )
    raw_df = parse_firms_csv_text(raw_text)
    return normalize_firms_nrt_dataframe(raw_df, product=req.product)


def observations_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a normalized observations DataFrame to a list of row dicts."""
    if df is None or df.empty:
        return []
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def summarize_observations(df: pd.DataFrame) -> dict[str, Any]:
    """
    Safe summary for manual scripts / logs (no secrets).

    Acquisition datetime is derived only for summary display when acq_date /
    acq_time are present; the returned observations DataFrame itself keeps
    native string fields for preprocessing.
    """
    summary: dict[str, Any] = {
        "observation_count": int(len(df)) if df is not None else 0,
        "columns": list(df.columns) if df is not None else [],
    }
    if df is None or df.empty:
        summary["latitude_range"] = None
        summary["longitude_range"] = None
        summary["acq_date_range"] = None
        return summary

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    summary["latitude_range"] = {
        "min": float(lat.min()) if lat.notna().any() else None,
        "max": float(lat.max()) if lat.notna().any() else None,
    }
    summary["longitude_range"] = {
        "min": float(lon.min()) if lon.notna().any() else None,
        "max": float(lon.max()) if lon.notna().any() else None,
    }
    if "acq_date" in df.columns:
        dates = df["acq_date"].dropna().astype(str)
        summary["acq_date_range"] = {
            "min": dates.min() if not dates.empty else None,
            "max": dates.max() if not dates.empty else None,
        }
    else:
        summary["acq_date_range"] = None
    return summary


__all__ = [
    "DEFAULT_VIIRS_NRT_PRODUCT",
    "REQUIRED_FIRMS_COLUMNS",
    "FirmsNRTConfigError",
    "FirmsNRTError",
    "FirmsNRTHttpError",
    "FirmsNRTParseError",
    "FirmsNRTRequest",
    "build_firms_area_url",
    "fetch_firms_nrt_csv",
    "fetch_firms_nrt_observations",
    "normalize_firms_nrt_dataframe",
    "observations_to_records",
    "parse_bbox",
    "parse_firms_csv_text",
    "redact_secrets",
    "summarize_observations",
    "validate_day_range",
    "validate_product",
]
