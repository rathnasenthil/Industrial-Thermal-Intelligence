"""Realtime event-formation configuration.

Reuses Stage G ``STDBSCANConfig`` spatial/temporal epsilons so incremental
matching stays consistent with the frozen batch clustering parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.event_formation.config import DEFAULT_CONFIG, STDBSCANConfig


@dataclass(frozen=True)
class RealtimeEventConfig:
    """
    Continuity thresholds for incremental event matching.

    Defaults are copied from ``STDBSCANConfig`` (spatial_eps_km=1.5,
    temporal_eps_hours=36.0). Do not invent unrelated thresholds.

    Note on min_samples: batch ST-DBSCAN requires min_samples=2 before a
    core point forms a cluster (singletons become noise). Streaming opens
    an event on the first observation and grows it; that is intentional
    for NRT state tracking and does **not** mean the observation is a
    confirmed fire.
    """

    spatial_eps_km: float = DEFAULT_CONFIG.spatial_eps_km
    temporal_eps_hours: float = DEFAULT_CONFIG.temporal_eps_hours

    @classmethod
    def from_stdbscan(cls, config: STDBSCANConfig | None = None) -> RealtimeEventConfig:
        cfg = config or DEFAULT_CONFIG
        return cls(
            spatial_eps_km=cfg.spatial_eps_km,
            temporal_eps_hours=cfg.temporal_eps_hours,
        )


def default_realtime_config() -> RealtimeEventConfig:
    return RealtimeEventConfig.from_stdbscan(DEFAULT_CONFIG)
