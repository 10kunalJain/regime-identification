/**
 * Shared types for the regime-identification dashboard.
 *
 * Mirrors `regime/api/schemas.py::RegimePosterior` and the static path payload
 * produced by `scripts/build_paper_inputs.py`.
 */

export interface RegimePosterior {
  as_of: string;
  regime_probs_uncal: Record<string, number>;
  crisis_prob_21d_cal: number;
  confidence: number;
  method: string;
  version: string;
}

export interface RegimePathPoint {
  data_time: string;
  state: number;
  crisis_prob: number;
  spy_close: number;
}

export type Freshness = "live" | "cached" | "static";
