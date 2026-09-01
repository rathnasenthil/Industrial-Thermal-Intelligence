/**
 * Placeholder domain types for thermal events detected from NASA FIRMS data.
 * TODO: Refine and extend once the backend event schema is finalized.
 */
export interface ThermalEvent {
  id: string;
  latitude: number;
  longitude: number;
  brightness: number;
  confidence: number;
  acquiredAt: string;
  classification?: ThermalEventClassification;
}

export type ThermalEventClassification =
  | "industrial_fire"
  | "wildfire"
  | "agricultural_burning"
  | "gas_flare"
  | "persistent_thermal_source"
  | "unknown";
