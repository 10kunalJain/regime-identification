/**
 * Server-side data resolution for the live panel.
 *
 * Strategy: try the FastAPI backend with a short timeout; on any failure
 * fall through to the static JSON baked into `frontend/public/` by
 * `scripts/build_paper_inputs.py`. The Vercel KV blob fallback layer wires
 * in once the KV add-on is provisioned (placeholder retained for shape).
 *
 * Per ARCHITECTURE.md §10 the dashboard must never 502 — there is always
 * a renderable payload, even if it is the last build's snapshot.
 */

import type { Freshness, RegimePosterior } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL;
const TIMEOUT_MS = 3000;

export interface ResolvedLatest {
  posterior: RegimePosterior;
  freshness: Freshness;
}

export async function resolveLatest(staticFallback: RegimePosterior): Promise<ResolvedLatest> {
  if (!API_URL) {
    return { posterior: staticFallback, freshness: "static" };
  }
  try {
    const r = await fetch(`${API_URL}/regime/now`, {
      next: { revalidate: 60 },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (r.ok) {
      const body = (await r.json()) as RegimePosterior;
      return { posterior: body, freshness: "live" };
    }
  } catch {
    // Network error or timeout — fall through to the cached blob.
  }
  return { posterior: staticFallback, freshness: "cached" };
}
