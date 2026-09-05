"""
Configuration for GIFT Stage I.5 (NASA STA Evidence Integration).

STA is supporting evidence only. Thresholds and layer priorities below are
ENGINEERING defaults — not scientifically validated causal distances or
classification rules. NASA FIRMS describes STA Mask / STA Detections as
experimental/provisional layers; this stage never treats them as ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Spatial relationship categories (event ↔ STA).
STA_INTERSECTS_EVENT = "STA_INTERSECTS_EVENT"
STA_NEAR_EVENT = "STA_NEAR_EVENT"
NO_STA_ASSOCIATION = "NO_STA_ASSOCIATION"

# Primary association status on the event row.
STA_ASSOCIATED = "STA_ASSOCIATED"
STA_AMBIGUOUS = "AMBIGUOUS"

# Layer types.
LAYER_MASK = "MASK"
LAYER_DETECTION = "DETECTION"

# Temporal relations.
TEMPORAL_SAME_PERIOD = "SAME_PERIOD"
TEMPORAL_NEAR_EVENT_TIME = "NEAR_EVENT_TIME"
TEMPORAL_OUTSIDE_EVENT_TIME = "OUTSIDE_EVENT_TIME"
TEMPORAL_NOT_APPLICABLE = "NOT_APPLICABLE"
TEMPORAL_UNKNOWN = "UNKNOWN"

# Evidence quality (match quality — not fire probability).
QUALITY_NONE = "NONE"
QUALITY_LOW = "LOW"
QUALITY_MEDIUM = "MEDIUM"
QUALITY_HIGH = "HIGH"

# Default documented NASA FIRMS STA provenance (no invented download endpoint).
DEFAULT_STA_SOURCE = "NASA_FIRMS"
DEFAULT_STA_SOURCE_URL = (
    "https://www.earthdata.nasa.gov/news/blog/firms-releases-new-features-identify-active-fires-type"
)
DEFAULT_STA_DOCUMENTATION_URL = (
    "https://wiki.earthdata.nasa.gov/spaces/FIRMS/blog/2025/02/28/425855667/"
    "FIRMS+incorporates+static+thermal+anomalies+data+to+help+users+differentiate+"
    "between+vegetation+and+non+vegetation+fires."
)

# Prefer MASK over DETECTION when ranking equally close candidates.
DEFAULT_LAYER_PRIORITY: dict[str, int] = {
    LAYER_MASK: 0,
    LAYER_DETECTION: 1,
}


@dataclass(frozen=True)
class STAConfig:
    """Tunable engineering parameters for Stage I.5.

    Attributes:
        association_radius_km: Maximum centroid-to-STA distance (km) for
            ``STA_NEAR_EVENT``. Engineering default 1.0 km — not a claim about
            STA spatial uncertainty.
        ambiguity_distance_tolerance_km: If the top two candidates share the
            same relationship tier and their distances differ by at most this
            amount, association is AMBIGUOUS.
        near_event_time_hours: Half-window for DETECTION timestamps vs event
            start/end to count as NEAR_EVENT_TIME.
        mask_path / detection_path: Optional local NASA STA files (GeoJSON,
            GPKG, Shapefile, or CSV with geometry). Neither is downloaded
            automatically — supply files under ``data/raw/`` after obtaining
            them from NASA FIRMS documentation / map export.
        events_path: Stage I.4 output (read-only).
    """

    association_radius_km: float = 1.0
    ambiguity_distance_tolerance_km: float = 0.1
    near_event_time_hours: float = 24.0
    max_candidates_per_event: int | None = 20
    layer_priority: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LAYER_PRIORITY))
    mask_path: Path | None = Path("data/raw/nasa_firms_sta_mask.geojson")
    detection_path: Path | None = Path("data/raw/nasa_firms_sta_detections.geojson")
    events_path: Path = Path("data/processed/thermal_events_with_anomaly_detection.csv")
    sta_source: str = DEFAULT_STA_SOURCE
    sta_source_url: str = DEFAULT_STA_SOURCE_URL
    sta_documentation_url: str = DEFAULT_STA_DOCUMENTATION_URL
    sta_source_version: str | None = None
    sta_download_date: str | None = None

    def __post_init__(self) -> None:
        if self.association_radius_km <= 0:
            raise ValueError("association_radius_km must be > 0.")
        if self.ambiguity_distance_tolerance_km < 0:
            raise ValueError("ambiguity_distance_tolerance_km must be >= 0.")
        if self.near_event_time_hours < 0:
            raise ValueError("near_event_time_hours must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mask_path"] = str(self.mask_path) if self.mask_path else None
        d["detection_path"] = str(self.detection_path) if self.detection_path else None
        d["events_path"] = str(self.events_path)
        return d

    def describe_rationale(self) -> dict[str, str]:
        return {
            "association_radius_km": (
                f"Engineering proximity threshold ({self.association_radius_km} km) for "
                "STA_NEAR_EVENT. Not a scientifically validated STA uncertainty radius."
            ),
            "ambiguity_distance_tolerance_km": (
                "When top candidates share the same spatial tier and distances within "
                "this tolerance, refuse to pick a single primary STA feature."
            ),
            "source_policy": (
                "No undocumented private NASA endpoint is hard-coded. Local STA Mask / "
                "Detection files must be supplied under data/raw/ after obtaining them "
                "from NASA FIRMS documentation. Fabricating STA geometries is forbidden."
            ),
            "independence_from_i4": (
                "I.5 appends STA evidence only. anomaly_score / anomaly_status / "
                "feature deviations from I.4 are never recalculated."
            ),
        }


DEFAULT_CONFIG = STAConfig()
