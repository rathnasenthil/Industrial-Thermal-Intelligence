import { useEffect, useState } from "react";
import { getHealth } from "../services/healthService";

export type HealthStatus = "idle" | "loading" | "online" | "offline";

/**
 * Small dev-time hook that pings the backend health endpoint on mount.
 * Useful for confirming the frontend/backend wiring during local setup.
 */
export function useHealthCheck(): HealthStatus {
  const [status, setStatus] = useState<HealthStatus>("idle");

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");

    getHealth()
      .then(() => {
        if (!cancelled) setStatus("online");
      })
      .catch(() => {
        if (!cancelled) setStatus("offline");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return status;
}
