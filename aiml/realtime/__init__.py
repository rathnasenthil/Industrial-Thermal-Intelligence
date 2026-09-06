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

Phase 4 adds incremental Stage G.1 persistence for the *affected event only*
(see ``realtime.persistence``). Persistence = repeatedly observed activity
over time — not confirmed fire, industrial source, or danger.

Phase 5 adds incremental Stage I.2 facility association for the *affected
event only* (see ``realtime.facility_association``). Association is spatial
attribution only — not source classification or industrial-fire confirmation.

Phase 6 adds incremental Stage I.3 facility fingerprinting for *affected
facilities only* (see ``realtime.facility_fingerprint``). Descriptive
baseline only — not anomaly detection.

Phase 7 adds incremental Stage I.4 temporal anomaly detection for the
*affected event only* (see ``realtime.anomaly``). Walk-forward prior-only
deviation — not risk scoring or fire classification. Does not use I.3
fingerprint tables as the scoring baseline.

Phase 8 adds incremental Stage I.5 NASA STA evidence for the *affected
event only* (see ``realtime.sta``). Supporting evidence only — not ground
truth or industrial-fire classification.

Phase 9 adds incremental Stage I.6 environmental / satellite context for
the *affected event only* (see ``realtime.environmental``). Context /
evidence only — not classification or risk scoring.

Phase 10 adds incremental Stage I.7 evidence fusion for the *affected
event only* (see ``realtime.evidence_fusion``). Interpretation only —
not ground truth or risk probability.

Phase 11 adds incremental Stage VI risk prioritization for the *affected
event only* (see ``realtime.risk``). Decision-support score — not fire
probability.

A thermal event here is a spatio-temporal cluster of FIRMS hotspot
observations — **not** a confirmed fire, industrial fire, or alert.
"""

from .anomaly import AnomalyResult, process_event_anomaly, unavailable_anomaly_result
from .config import RealtimeEventConfig, default_realtime_config
from .environmental import (
    EnvironmentalContextResult,
    process_event_environmental,
    unavailable_environmental_result,
)
from .evidence_fusion import EvidenceFusionResult, process_event_evidence_fusion
from .facility_association import AssociationResult, process_event_facility_association
from .facility_fingerprint import FacilityFingerprintResult, process_facility_fingerprint
from .incremental_processor import process_observation
from .persistence import PersistenceFeatures, process_event_persistence
from .risk import RiskPrioritizationResult, process_event_risk
from .schemas import ActiveEventState, MatchAction, ObservationRecord, ProcessResult
from .sta import STAEvidenceResult, process_event_sta, unavailable_sta_result

__all__ = [
    "ActiveEventState",
    "AnomalyResult",
    "AssociationResult",
    "EnvironmentalContextResult",
    "EvidenceFusionResult",
    "FacilityFingerprintResult",
    "MatchAction",
    "ObservationRecord",
    "PersistenceFeatures",
    "ProcessResult",
    "RealtimeEventConfig",
    "RiskPrioritizationResult",
    "STAEvidenceResult",
    "default_realtime_config",
    "process_event_anomaly",
    "process_event_environmental",
    "process_event_evidence_fusion",
    "process_event_facility_association",
    "process_event_persistence",
    "process_event_risk",
    "process_event_sta",
    "process_facility_fingerprint",
    "process_observation",
    "unavailable_anomaly_result",
    "unavailable_environmental_result",
    "unavailable_sta_result",
]
