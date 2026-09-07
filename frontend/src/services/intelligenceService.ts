import { apiClient } from "./apiClient";
import type {
  DashboardStatistics,
  EventDetail,
  EventEvidence,
  EventSummary,
  EventTimeline,
  Facility,
  FacilityDetail,
  Pagination,
} from "../types/api";

export interface EventFilters {
  page?: number;
  page_size?: number;
  priority?: string;
  industrial_context?: string;
  facility_type?: string;
  persistence_class?: string;
  anomaly_status?: string;
  date_from?: string;
  date_to?: string;
  min_risk_score?: number;
  max_risk_score?: number;
  bbox?: string;
}

const query = (filters: EventFilters) => {
  const params = Object.fromEntries(
    Object.entries(filters).filter(([, value]) => value !== undefined && value !== ""),
  );
  return { params };
};

export async function getDashboardStatistics() {
  const { data } = await apiClient.get<DashboardStatistics>("/dashboard/statistics");
  return data;
}

export async function getEvents(filters: EventFilters = {}) {
  const { data } = await apiClient.get<Pagination<EventSummary>>("/events", query(filters));
  return data;
}

export async function getAlerts(filters: EventFilters = {}) {
  const { data } = await apiClient.get<Pagination<EventSummary>>("/alerts", query(filters));
  return data;
}

export async function getEvent(eventId: string) {
  const { data } = await apiClient.get<EventDetail>(`/events/${encodeURIComponent(eventId)}`);
  return data;
}

export async function getEventEvidence(eventId: string) {
  const { data } = await apiClient.get<EventEvidence>(`/events/${encodeURIComponent(eventId)}/evidence`);
  return data;
}

export async function getEventTimeline(eventId: string) {
  const { data } = await apiClient.get<EventTimeline>(`/events/${encodeURIComponent(eventId)}/timeline`);
  return data;
}

export interface FacilityFilters {
  page?: number;
  page_size?: number;
  facility_type?: string;
  search?: string;
  bbox?: string;
}

export async function getFacilities(filters: FacilityFilters = {}) {
  const { data } = await apiClient.get<Pagination<Facility>>("/facilities", {
    params: Object.fromEntries(Object.entries(filters).filter(([, value]) => value !== undefined && value !== "")),
  });
  return data;
}

export async function getFacility(facilityId: string) {
  const { data } = await apiClient.get<FacilityDetail>(`/facilities/${encodeURIComponent(facilityId)}`);
  return data;
}

export async function getFacilityHistory(facilityId: string, page = 1) {
  const { data } = await apiClient.get<Pagination<EventSummary>>(
    `/facilities/${encodeURIComponent(facilityId)}/history`,
    { params: { page, page_size: 50 } },
  );
  return data;
}
