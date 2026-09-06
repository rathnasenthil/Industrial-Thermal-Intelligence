"""
Near-real-time incremental thermal-event formation (AIML side).

This package implements streaming-compatible event matching and aggregate
updates. It deliberately does **not** re-run full-batch ST-DBSCAN:

- Batch ``_assign_event_ids`` renumbers ``EVT_#######`` by earliest cluster
  time across the *entire* dataset. Adding an earlier observation would
  renumber historical IDs — unacceptable for streaming.
- Batch ST-DBSCAN builds a global neighbor graph; incremental processing
  attaches each new observation to an *existing active event* (or opens a
  new one) using the same spatial/temporal continuity parameters from
  ``STDBSCANConfig``.

A thermal event here is a spatio-temporal cluster of FIRMS hotspot
observations — **not** a confirmed fire, industrial fire, or alert.
"""

from .config import RealtimeEventConfig, default_realtime_config
from .incremental_processor import process_observation
from .schemas import ActiveEventState, MatchAction, ObservationRecord, ProcessResult

__all__ = [
    "ActiveEventState",
    "MatchAction",
    "ObservationRecord",
    "ProcessResult",
    "RealtimeEventConfig",
    "default_realtime_config",
    "process_observation",
]
