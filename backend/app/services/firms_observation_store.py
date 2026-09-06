"""
Persist normalized FIRMS NRT observations with hash-based deduplication.

Phase 2 only: store observations in ``firms_observations``. Does not assign
``event_id``, run AIML stages, schedule polls, or truncate existing tables.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from geoalchemy2.elements import WKTElement
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.firms_observation import FirmsObservation
from app.services.firms_nrt_ingestion import (
    FirmsNRTError,
    fetch_firms_nrt_observations,
)
from app.services.observation_identity import compute_observation_hash

logger = logging.getLogger(__name__)

BATCH_SIZE = 1_000


class FirmsObservationStoreError(Exception):
    """Raised when an observation cannot be stored (e.g. invalid coordinates)."""


@dataclass
class StoreResult:
    received: int = 0
    inserted: int = 0
    duplicates: int = 0
    rejected: int = 0
    rejection_samples: list[str] = field(default_factory=list)
    inserted_hashes: list[str] = field(default_factory=list)

    def add_rejection(self, message: str, *, limit: int = 25) -> None:
        if len(self.rejection_samples) < limit:
            self.rejection_samples.append(message)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Keep manual/script output compact unless needed.
        if len(payload.get("inserted_hashes", [])) > 20:
            payload["inserted_hashes_sample"] = payload["inserted_hashes"][:20]
            payload["inserted_hashes_count"] = len(payload["inserted_hashes"])
            del payload["inserted_hashes"]
        return payload


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _optional_str(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    return str(value).strip()


def _optional_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def parse_acq_datetime(
    acq_date: Optional[str], acq_time: Optional[str]
) -> Optional[datetime]:
    """
    Best-effort UTC datetime from FIRMS acq_date + acq_time.

    Supports YYYY-MM-DD dates and HHMM / HMM times. Returns None if unparseable
    (observation can still be stored; datetime is nullable).
    """
    if not acq_date:
        return None
    date_text = acq_date.strip()
    time_text = (acq_time or "0").strip()
    if time_text.isdigit():
        time_text = time_text.zfill(4)
    try:
        if len(time_text) == 4 and time_text.isdigit():
            hours = int(time_text[:2])
            minutes = int(time_text[2:])
            dt = datetime.strptime(date_text, "%Y-%m-%d").replace(
                hour=hours, minute=minutes, tzinfo=timezone.utc
            )
            return dt
        # Fallback: date only at 00:00 UTC
        return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def validate_coordinates(lat: Any, lon: Any) -> tuple[float, float]:
    """Validate and return (latitude, longitude). Raises on invalid values."""
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError) as exc:
        raise FirmsObservationStoreError(
            f"Non-numeric coordinates: latitude={lat!r}, longitude={lon!r}"
        ) from exc
    if math.isnan(latitude) or math.isnan(longitude):
        raise FirmsObservationStoreError("Coordinates are NaN")
    if not (-90.0 <= latitude <= 90.0):
        raise FirmsObservationStoreError(
            f"latitude out of range [-90, 90]: {latitude}"
        )
    if not (-180.0 <= longitude <= 180.0):
        raise FirmsObservationStoreError(
            f"longitude out of range [-180, 180]: {longitude}"
        )
    return latitude, longitude


def observation_row_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert a normalized FIRMS row into a DB insert dict.

    ``event_id`` is always NULL in Phase 2.
    """
    latitude, longitude = validate_coordinates(raw.get("latitude"), raw.get("longitude"))
    obs_hash = compute_observation_hash(raw)
    acq_date = _optional_str(raw.get("acq_date"))
    acq_time = _optional_str(raw.get("acq_time"))
    return {
        "observation_hash": obs_hash,
        "latitude": latitude,
        "longitude": longitude,
        "geometry": WKTElement(f"POINT({longitude} {latitude})", srid=4326),
        "acq_date": acq_date,
        "acq_time": acq_time,
        "acq_datetime": parse_acq_datetime(acq_date, acq_time),
        "satellite": _optional_str(raw.get("satellite")),
        "instrument": _optional_str(raw.get("instrument")),
        "confidence": _optional_str(raw.get("confidence")),
        "version": _optional_str(raw.get("version")),
        "bright_ti4": _optional_float(raw.get("bright_ti4")),
        "bright_ti5": _optional_float(raw.get("bright_ti5")),
        "scan": _optional_float(raw.get("scan")),
        "track": _optional_float(raw.get("track")),
        "frp": _optional_float(raw.get("frp")),
        "daynight": _optional_str(raw.get("daynight")),
        "type": _optional_str(raw.get("type")),
        "source_file": _optional_str(raw.get("source_file")),
        "event_id": None,
    }


def store_firms_observations(
    session: Session,
    observations: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    commit: bool = True,
) -> StoreResult:
    """
    Insert observations with ON CONFLICT DO NOTHING on observation_hash.

    Returns insert/duplicate/reject statistics. Does not truncate the table.
    """
    result = StoreResult()
    if observations is None:
        return result

    if isinstance(observations, pd.DataFrame):
        records: list[Mapping[str, Any]] = observations.to_dict(orient="records")
    else:
        records = list(observations)

    result.received = len(records)
    batch: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        batch_hashes = [row["observation_hash"] for row in batch]
        stmt = (
            pg_insert(FirmsObservation.__table__)
            .values(batch)
            .on_conflict_do_nothing(index_elements=["observation_hash"])
            .returning(FirmsObservation.__table__.c.observation_hash)
        )
        inserted_hashes = list(session.execute(stmt).scalars().all())
        inserted = len(inserted_hashes)
        result.inserted += inserted
        result.duplicates += len(batch) - inserted
        result.inserted_hashes.extend(inserted_hashes)
        batch = []
        _ = batch_hashes  # retained for clarity / future diagnostics

    for raw in records:
        try:
            row = observation_row_from_mapping(raw)
        except FirmsObservationStoreError as exc:
            result.rejected += 1
            result.add_rejection(str(exc))
            continue

        obs_hash = row["observation_hash"]
        if obs_hash in seen_hashes:
            # Duplicate within this batch/payload — count as duplicate, skip.
            result.duplicates += 1
            continue
        seen_hashes.add(obs_hash)
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            flush()

    flush()
    if commit:
        session.commit()
    else:
        session.flush()

    logger.info(
        "FIRMS observation store: received=%s inserted=%s duplicates=%s rejected=%s",
        result.received,
        result.inserted,
        result.duplicates,
        result.rejected,
    )
    return result


def fetch_and_store_firms_nrt(
    session: Session,
    *,
    settings: Optional[Settings] = None,
    commit: bool = True,
    **fetch_kwargs: Any,
) -> tuple[pd.DataFrame, StoreResult]:
    """
    Phase 1 fetch + Phase 2 store.

    Keeps the HTTP client independently usable via ``fetch_firms_nrt_observations``.
    """
    cfg = settings or get_settings()
    try:
        df = fetch_firms_nrt_observations(settings=cfg, **fetch_kwargs)
    except FirmsNRTError:
        raise
    store_result = store_firms_observations(session, df, commit=commit)
    return df, store_result
