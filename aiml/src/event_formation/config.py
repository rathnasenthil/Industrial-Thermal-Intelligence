"""
Configuration for ST-DBSCAN-based Geospatial Event Formation (GIFT Stage G).

Every parameter below is an ENGINEERING default chosen from basic domain
reasoning about VIIRS NOAA-20 overpass geometry and pixel footprint size
(see the rationale in each docstring / comment). None of them are
scientifically validated thresholds — they have not been tuned against
labeled fire-event ground truth. They are intentionally exposed here (and
recorded in every run's report) so they can be revisited and tuned later
without touching the clustering code itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class STDBSCANConfig:
    """Tunable parameters for spatio-temporal thermal-event clustering.

    Attributes:
        spatial_eps_km: Maximum great-circle distance (km) between two
            detections for them to be considered spatial neighbors.

            Rationale for the default (1.5 km): the actual FIRMS VIIRS
            NOAA-20 records in this project report per-detection ``scan``
            and ``track`` pixel-size values (observed range ~0.32-0.80 km
            in aiml/data/processed/firms_viirs_india_2023_2024_clean.csv).
            VIIRS I-band pixels are ~0.375 km at nadir and grow toward the
            edge of the scan (the "bowtie" effect) — the observed
            0.32-0.80 km range in this dataset reflects exactly that.
            1.5 km is roughly ~2x the largest observed single-pixel
            footprint, chosen to link genuinely adjacent pixels/overlapping
            footprints (including geolocation jitter) while not reaching
            far enough to casually bridge unrelated, merely nearby sources.
            It is deliberately NOT presented as "the" VIIRS footprint size
            — no single distance can be, given how much scan/track vary
            with scan angle.
        temporal_eps_hours: Maximum time difference (hours) between two
            detections for them to be considered temporal neighbors.

            Rationale for the default (36 hours): this dataset shows NOAA-20
            passing over India roughly twice a day, clustered around
            ~06:00-09:00 UTC (daytime) and ~18:00-22:00 UTC (nighttime) —
            i.e. about 12 hours apart. 36 hours allows a persistent thermal
            source to be linked across one missed overpass (e.g. due to
            cloud cover) without stretching far enough to casually connect
            unrelated activity that happens to occur days apart at a
            similar location.
        min_samples: Minimum number of detections (including the point
            itself) required within `spatial_eps_km` and
            `temporal_eps_hours` for a detection to be a DBSCAN "core"
            point.

            Rationale for the default (2): requires at least one
            corroborating detection near a given one, in both space and
            time, before treating them as an "event" rather than an
            unconfirmed single detection. Single, uncorroborated
            detections are NOT deleted — they are written to
            `thermal_event_noise.csv` for later stages to still make use
            of individually.
        query_batch_size: Number of detections processed per spatial index
            query batch. This bounds peak memory (each batch only holds
            that many detections' neighbor-index arrays at once) and has
            no effect on clustering results — see
            `src.event_formation.st_dbscan` for why batching cannot split
            events at "boundaries" (there are none; the spatial index
            always covers the full dataset).
    """

    spatial_eps_km: float = 1.5
    temporal_eps_hours: float = 36.0
    min_samples: int = 2
    query_batch_size: int = 20_000

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain, JSON-serializable dict."""
        return asdict(self)

    def describe_rationale(self) -> dict[str, str]:
        """Human-readable rationale strings for each parameter (for reports)."""
        return {
            "spatial_eps_km": (
                "~2x the largest observed VIIRS pixel footprint (scan/track "
                "0.32-0.80 km in the processed dataset) to link adjacent/"
                "overlapping pixel detections; not a claim about a single "
                "universal VIIRS footprint size, since footprint grows "
                "with scan angle."
            ),
            "temporal_eps_hours": (
                "~3x the observed ~12h NOAA-20 day/night overpass interval "
                "over India, allowing one missed overpass (e.g. cloud "
                "cover) to still be bridged for a persistent source."
            ),
            "min_samples": (
                "requires at least one corroborating nearby detection "
                "before calling a group of detections a 'thermal event'; "
                "uncorroborated single detections are preserved as noise, "
                "not deleted."
            ),
            "query_batch_size": (
                "engineering parameter only, bounds peak memory during "
                "spatial-index queries; does not affect clustering output."
            ),
        }


DEFAULT_CONFIG = STDBSCANConfig()
