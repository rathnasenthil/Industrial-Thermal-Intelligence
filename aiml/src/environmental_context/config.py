"""
Configuration for GIFT Stage I.6 (Environmental / Satellite Context).

All buffer distances and expected dataset paths are ENGINEERING defaults.
They are not scientifically validated causal radii. Datasets are never
downloaded or fabricated — missing files yield explicit unavailable
evidence, not zeros.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentalContextConfig:
    """Tunable parameters and local dataset path expectations for I.6.

    Attributes:
        context_buffer_km: Local context buffer around the event footprint
            (engineering default 1.0 km). Used for vector intersection /
            coverage summaries when datasets exist.
        broad_context_buffer_km: Broader search radius for nearest-feature
            distance queries (engineering default 5.0 km).
        events_path: Preferred Stage I.5 events output; falls back to I.4
            when the I.5 events CSV was not produced (e.g. STA source missing).
        landcover_raster_path / landcover_vector_path: Optional local
            land-cover sources.
        vegetation_path, builtup_path, water_path, agriculture_path:
            Optional local vector layers.
        satellite_raster_path: Optional local satellite-derived raster.
        landcover_class_map: Optional mapping from source class codes to
            human-readable labels. Empty means use raw class IDs as strings.
    """

    context_buffer_km: float = 1.0
    broad_context_buffer_km: float = 5.0
    events_path: Path = Path("data/processed/thermal_events_with_sta_evidence.csv")
    events_fallback_path: Path = Path("data/processed/thermal_events_with_anomaly_detection.csv")
    landcover_raster_path: Path | None = Path("data/external/landcover.tif")
    landcover_vector_path: Path | None = Path("data/external/landcover.geojson")
    vegetation_path: Path | None = Path("data/external/vegetation.geojson")
    builtup_path: Path | None = Path("data/external/builtup.geojson")
    water_path: Path | None = Path("data/external/water.geojson")
    agriculture_path: Path | None = Path("data/external/agriculture.geojson")
    satellite_raster_path: Path | None = Path("data/external/satellite_context.tif")
    landcover_class_map: dict[str, str] = field(default_factory=dict)
    landcover_year: str | None = None
    landcover_source_name: str = "local_landcover"

    def __post_init__(self) -> None:
        if self.context_buffer_km <= 0:
            raise ValueError("context_buffer_km must be > 0.")
        if self.broad_context_buffer_km < self.context_buffer_km:
            raise ValueError("broad_context_buffer_km must be >= context_buffer_km.")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in (
            "events_path",
            "events_fallback_path",
            "landcover_raster_path",
            "landcover_vector_path",
            "vegetation_path",
            "builtup_path",
            "water_path",
            "agriculture_path",
            "satellite_raster_path",
        ):
            val = d[key]
            d[key] = str(val) if val is not None else None
        return d

    def describe_rationale(self) -> dict[str, str]:
        return {
            "context_buffer_km": (
                f"Engineering local buffer ({self.context_buffer_km} km) around the "
                "Stage G detection envelope for coverage summaries — not a validated "
                "ecological or fire-spread radius."
            ),
            "broad_context_buffer_km": (
                f"Engineering search radius ({self.broad_context_buffer_km} km) for "
                "nearest-feature distances when local intersection finds nothing."
            ),
            "missing_data": (
                "Absent datasets produce availability=false and null evidence fields. "
                "Missing evidence is never coerced to 0."
            ),
            "no_classification": (
                "I.6 produces environmental context evidence only. It does not assign "
                "industrial/wildfire/agricultural source labels or risk scores."
            ),
        }


DEFAULT_CONFIG = EnvironmentalContextConfig()
