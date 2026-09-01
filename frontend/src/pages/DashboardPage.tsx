import { useHealthCheck, type HealthStatus } from "../hooks/useHealthCheck";

const statusLabel: Record<HealthStatus, string> = {
  idle: "Idle",
  loading: "Checking backend...",
  online: "Backend online",
  offline: "Backend unreachable",
};

const statusColor: Record<HealthStatus, string> = {
  idle: "bg-slate-500",
  loading: "bg-yellow-500",
  online: "bg-green-500",
  offline: "bg-red-500",
};

/**
 * Placeholder landing page.
 * TODO: Replace with the GIS map, event feed, and dashboard widgets.
 */
export function DashboardPage() {
  const status = useHealthCheck();

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <section className="rounded-lg border border-slate-800 bg-slate-900 p-6">
        <h2 className="text-xl font-medium">Environment ready</h2>
        <p className="mt-2 text-sm text-slate-400">
          This is a placeholder dashboard confirming the frontend scaffold
          builds and runs. The GIS map, thermal event feed, alerts panel and
          analytics widgets will be implemented here.
        </p>
      </section>

      <section className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-900 p-6">
        <span className={`h-2.5 w-2.5 rounded-full ${statusColor[status]}`} />
        <span className="text-sm text-slate-300">{statusLabel[status]}</span>
      </section>
    </div>
  );
}
