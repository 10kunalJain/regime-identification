/**
 * Typed client for the FastAPI backend.
 *
 * Uses a 5s timeout and falls back to the Vercel KV cached blob when the
 * upstream API is unreachable. Per ARCHITECTURE.md §10, this is the
 * resilience mechanism that keeps the dashboard up during Oracle outages.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const KV_FALLBACK_KEY = "regime:latest";

export interface RegimePosterior {
  as_of: string;
  regime_probs_uncal: Record<string, number>;
  crisis_prob_21d_cal: number;
  confidence: number;
  method: string;
  version: string;
}

export async function fetchLatestRegime(): Promise<RegimePosterior | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const r = await fetch(`${API_URL}/regime/now`, {
      signal: controller.signal,
      next: { revalidate: 60 },
    });
    if (r.ok) {
      const body = (await r.json()) as RegimePosterior;
      // Best-effort cache to KV; ignore errors so a KV outage doesn't break the read path.
      void cacheToKv(body);
      return body;
    }
  } catch {
    // fall through to KV
  } finally {
    clearTimeout(timeout);
  }
  return await readFromKv();
}

async function cacheToKv(value: RegimePosterior): Promise<void> {
  // Placeholder. Vercel KV client is wired in at `lib/kv.ts` once the
  // KV add-on is provisioned for the project (Vercel free tier allows 1 KV).
  void value;
}

async function readFromKv(): Promise<RegimePosterior | null> {
  // Placeholder. Returns null when KV is not yet provisioned; the page renders
  // the "no posterior" state in that case.
  void KV_FALLBACK_KEY;
  return null;
}
