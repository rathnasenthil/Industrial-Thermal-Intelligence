/**
 * Centralised access to build-time environment variables.
 * All Vite env vars consumed by the app must be re-exported from here so
 * that the rest of the codebase never reads `import.meta.env` directly.
 */
export const env = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api",
} as const;
