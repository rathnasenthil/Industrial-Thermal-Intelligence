import { apiClient } from "./apiClient";

export interface HealthResponse {
  status: string;
}

/**
 * Calls the backend `/health` endpoint.
 * Used to verify frontend <-> backend connectivity during development.
 */
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/health");
  return data;
}
