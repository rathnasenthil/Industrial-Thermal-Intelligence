/**
 * Formatting helpers shared across the dashboard.
 * TODO: Expand with date/coordinate/thermal-value formatters as pages are built.
 */
export function formatCoordinate(value: number, precision = 4): string {
  return value.toFixed(precision);
}
