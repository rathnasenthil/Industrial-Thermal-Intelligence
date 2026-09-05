"""Discover and load independent validation reference datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.validation.config import FORBIDDEN_PSEUDO_LABEL_SOURCES, ValidationConfig
from src.validation.label_normalization import normalize_reference_labels
from src.validation.validation_schema import clean_text, empty_canonical_frame


def discover_validation_paths(config: ValidationConfig) -> list[Path]:
    """Find candidate validation files by configured hints (local only)."""
    found: list[Path] = []
    primary = Path(config.validation_path)
    if primary.exists() and primary.is_file():
        found.append(primary)

    for directory in config.validation_search_dirs:
        d = Path(directory)
        if not d.exists() or not d.is_dir():
            continue
        for path in sorted(d.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".csv", ".geojson", ".json", ".gpkg", ".parquet"}:
                continue
            name = path.name.lower()
            if any(hint in name for hint in config.validation_filename_hints):
                if path not in found:
                    found.append(path)
    return found


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".geojson", ".json", ".gpkg"}:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        if gdf.geometry is not None:
            df["reference_geometry_wkt"] = gdf.geometry.to_wkt()
            cents = gdf.geometry.centroid
            if "reference_latitude" not in df.columns:
                df["reference_latitude"] = cents.y
            if "reference_longitude" not in df.columns:
                df["reference_longitude"] = cents.x
        return df
    raise ValueError(f"Unsupported validation file type: {path}")


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common aliases into the canonical schema."""
    aliases = {
        "validation_id": ["validation_id", "id", "ref_id", "reference_id"],
        "event_id": ["event_id", "thermal_event_id"],
        "reference_label_raw": [
            "reference_label_raw",
            "reference_label",
            "label",
            "label_raw",
            "fire_type",
            "source_type",
        ],
        "reference_source": ["reference_source", "source", "dataset", "provenance"],
        "reference_date": ["reference_date", "date", "acq_date", "incident_date", "event_date"],
        "reference_latitude": ["reference_latitude", "latitude", "lat", "y"],
        "reference_longitude": ["reference_longitude", "longitude", "lon", "lng", "x"],
        "reference_geometry_wkt": ["reference_geometry_wkt", "geometry_wkt", "wkt"],
        "reference_confidence": ["reference_confidence", "confidence", "label_confidence"],
        "label_notes": ["label_notes", "notes", "comment"],
        "validation_source": ["validation_source", "source_name"],
        "validation_source_independent": [
            "validation_source_independent",
            "independent",
            "is_independent",
        ],
        "validation_label_verified": ["validation_label_verified", "verified"],
    }
    out = pd.DataFrame(index=df.index)
    lower_map = {c.lower(): c for c in df.columns}
    for canonical, names in aliases.items():
        chosen = None
        for name in names:
            if name.lower() in lower_map:
                chosen = lower_map[name.lower()]
                break
        out[canonical] = df[chosen] if chosen is not None else None
    return out


def assess_independence(source: str | None, config: ValidationConfig) -> bool:
    """Return False for known pipeline-derived / circular sources."""
    text = (source or "").strip().lower()
    if not text:
        return False
    if any(bad in text for bad in FORBIDDEN_PSEUDO_LABEL_SOURCES):
        return False
    # Explicit allow markers
    if "independent" in text or "curated" in text or "official" in text or "manual" in text:
        return True
    # Unknown sources are not assumed independent when required.
    if config.require_independent_source:
        return False
    return True


def load_validation_dataset(
    path: str | Path | None,
    config: ValidationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize a validation reference table.

    Returns empty canonical frame + meta when path is missing.
    """
    meta: dict[str, Any] = {
        "available": False,
        "path": str(path) if path is not None else None,
        "record_count": 0,
        "warnings": [],
    }
    if path is None:
        meta["warnings"].append("No validation path provided.")
        return empty_canonical_frame(), meta

    p = Path(path)
    if not p.exists():
        meta["warnings"].append(f"Validation file not found: {p}")
        return empty_canonical_frame(), meta

    raw = _read_tabular(p)
    mapped = _map_columns(raw)
    n = len(mapped)
    if mapped["validation_id"].isna().all():
        mapped["validation_id"] = [f"val_{i:06d}" for i in range(n)]
    else:
        mapped["validation_id"] = [
            clean_text(v, f"val_{i:06d}") for i, v in enumerate(mapped["validation_id"].tolist())
        ]

    mapped["reference_source"] = mapped["reference_source"].map(lambda v: clean_text(v))
    mapped["validation_source"] = mapped["validation_source"].map(
        lambda v: clean_text(v, clean_text(mapped["reference_source"].iloc[0] if n else None, p.name))
    )
    # Per-row independence
    indep = []
    for src, flag in zip(
        mapped["reference_source"].tolist(),
        mapped["validation_source_independent"].tolist(),
    ):
        if flag is not None and str(flag).strip().lower() in {"true", "1", "yes"}:
            indep.append(True)
        elif flag is not None and str(flag).strip().lower() in {"false", "0", "no"}:
            indep.append(False)
        else:
            indep.append(assess_independence(src if src else mapped["validation_source"].iloc[0], config))
    mapped["validation_source_independent"] = indep

    if "validation_label_verified" not in mapped or mapped["validation_label_verified"].isna().all():
        mapped["validation_label_verified"] = False
    else:
        mapped["validation_label_verified"] = (
            mapped["validation_label_verified"]
            .map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})
            .fillna(False)
        )

    mapped = normalize_reference_labels(mapped)
    mapped["validation_match_status"] = None
    mapped["reference_geometry_wkt"] = mapped.get("reference_geometry_wkt")
    for col in (
        "event_id",
        "reference_date",
        "reference_latitude",
        "reference_longitude",
        "reference_confidence",
        "label_notes",
    ):
        if col not in mapped.columns:
            mapped[col] = None

    from src.validation.validation_schema import CANONICAL_COLUMNS

    out = mapped.reindex(columns=list(CANONICAL_COLUMNS))
    meta["available"] = True
    meta["record_count"] = int(len(out))
    meta["path"] = str(p)
    if not any(out["validation_source_independent"].tolist()):
        meta["warnings"].append(
            "No records marked as independent sources under configured criteria."
        )
    return out, meta
