import axios from "axios";
import { env } from "../config/env";

/**
 * Shared Axios instance for all backend API calls.
 * Feature-specific services (e.g. events, alerts) should import this
 * client instead of creating their own Axios instances.
 */
export const apiClient = axios.create({
  baseURL: env.apiBaseUrl,
  timeout: 15000,
});
