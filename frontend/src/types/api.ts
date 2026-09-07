export interface Pagination<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface PointGeometry {
  type: "Point" | string;
  coordinates: [number, number] | number[];
}

export interface EventSummary {
  event_id: string;
  event_start: string | null;
  event_end: string | null;
  observed_duration_hours: number | null;
  detection_count: number | null;
  peak_frp: number | null;
  mean_frp: number | null;
  latitude: number | null;
  longitude: number | null;
  geometry: PointGeometry | null;
  persistence_label: string | null;
  facility_id: string | null;
  facility_name: string | null;
  facility_type: string | null;
  facility_association_method: string | null;
  facility_distance_km: number | null;
  anomaly_status: string | null;
  industrial_context: string | null;
  risk_score: number | null;
  investigation_priority: string | null;
  thermal_severity_band: string | null;
  recommended_action: string | null;
}

export interface FacilityCandidate {
  facility_id: string;
  facility_name: string | null;
  facility_type: string | null;
  spatial_relation: string | null;
  distance_km: number | null;
  candidate_rank: number | null;
  candidate_score: number | null;
}

export interface EventDetail extends EventSummary {
  distinct_detection_days: number | null;
  span_days: number | null;
  duty_cycle: number | null;
  mean_gap_hours: number | null;
  max_gap_hours: number | null;
  median_frp: number | null;
  total_frp: number | null;
  day_detection_count: number | null;
  night_detection_count: number | null;
  min_latitude: number | null;
  max_latitude: number | null;
  min_longitude: number | null;
  max_longitude: number | null;
  centroid_wkt: string | null;
  footprint_wkt: string | null;
  persistence_basis: string | null;
  facility_attribution_confidence: string | null;
  candidate_facility_count: number | null;
  facility_candidates: FacilityCandidate[];
  baseline_observation_count: number | null;
  baseline_history_status: string | null;
  anomaly_unavailable_reason: string | null;
  anomaly_score: number | null;
  anomaly_confidence: string | null;
  anomaly_explanation: string | null;
  evidence_sufficiency: string | null;
  evidence_uncertainty: string | null;
  evidence_strength: string | null;
  industrial_evidence_score: number | null;
  evidence_fusion_score: number | null;
  source_intelligence_candidate: string | null;
  candidate_rationale: string | null;
  candidate_is_ground_truth: boolean | null;
  interpretation_confidence: string | null;
  thermal_severity_score: number | null;
  uncertainty_score: number | null;
  uncertainty_band: string | null;
  dominant_risk_factors: string | null;
  dominant_uncertainty_factors: string | null;
  priority_reasons: string | null;
  priority_warnings: string | null;
  risk_limiting_evidence_codes: string | null;
  risk_scoring_version: string | null;
  semantics_note: string;
}

export interface EvidenceFamily {
  available: boolean;
  status: string;
  score: number | null;
  summary: string | null;
  details: Record<string, unknown>;
}

export interface EventEvidence {
  event_id: string;
  temporal: EvidenceFamily;
  infrastructure: EvidenceFamily;
  historical: EvidenceFamily;
  anomaly: EvidenceFamily;
  sta: EvidenceFamily;
  environmental: EvidenceFamily;
  fusion: Record<string, unknown>;
}

export interface EventTimeline {
  event_id: string;
  event_start: string | null;
  event_end: string | null;
  observed_duration_hours: number | null;
  distinct_detection_days: number | null;
  span_days: number | null;
  duty_cycle: number | null;
  mean_gap_hours: number | null;
  max_gap_hours: number | null;
  detection_count: number | null;
  day_detection_count: number | null;
  night_detection_count: number | null;
  detection_level_timeline_available: boolean;
  detection_level_timeline_note: string;
}

export interface Facility {
  facility_id: string;
  facility_name: string | null;
  facility_type: string | null;
  latitude: number | null;
  longitude: number | null;
  geometry: PointGeometry | null;
  osm_id: string | null;
  osm_type: string | null;
  confidence: string | null;
}

export interface FacilityDetail extends Facility {
  industrial_subtype: string | null;
  operator: string | null;
  landuse: string | null;
  power_type: string | null;
  man_made_type: string | null;
  geometry_type: string | null;
  geometry_wkt: string | null;
  osm_tags: Record<string, unknown> | null;
  source: string | null;
  source_version: string | null;
  thermal_summary: {
    associated_event_count: number;
    high_priority_count: number;
    critical_count: number;
    max_risk_score: number | null;
    latest_event_start: string | null;
  };
  semantics_note: string;
}

export interface DashboardStatistics {
  total_events: number;
  total_facilities: number;
  priority_distribution: Record<string, number>;
  industrial_context_distribution: Record<string, number>;
  persistence_distribution: Record<string, number>;
  thermal_severity_distribution: Record<string, number>;
  anomaly_distribution: Record<string, number>;
  facility_type_distribution: Record<string, number>;
  events_with_facility_association: number;
  events_without_facility_association: number;
  high_priority_count: number;
  critical_count: number;
  date_range_start: string | null;
  date_range_end: string | null;
  semantics_note: string;
}
