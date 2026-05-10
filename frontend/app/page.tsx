import { fetchLatestRegime } from "@/lib/api";

// ISR with 60s revalidation per ARCHITECTURE.md §10. The Vercel KV cached
// blob is used as a fallback when the API is unreachable, ensuring the
// dashboard never returns 502 even if the Oracle backend is down.
export const revalidate = 60;

export default async function HomePage() {
  const posterior = await fetchLatestRegime();

  return (
    <main className="min-h-screen px-6 py-10">
      <header className="mb-8">
        <h1 className="font-mono text-2xl font-semibold tracking-tight">
          Regime Identification
        </h1>
        <p className="mt-1 text-sm text-neutral-600">
          Filtered-only fair-evaluation benchmark — live dashboard.
        </p>
      </header>

      <section aria-labelledby="live-regime" className="mb-12">
        <h2 id="live-regime" className="font-mono text-lg font-semibold">
          Live regime
        </h2>
        <div className="mt-4 rounded border border-neutral-200 p-4 font-mono text-sm">
          {posterior ? (
            <pre className="whitespace-pre-wrap">
              {JSON.stringify(posterior, null, 2)}
            </pre>
          ) : (
            <span className="text-neutral-500">
              No posterior available yet — start the FastAPI backend and
              ingest at least one observation.
            </span>
          )}
        </div>
      </section>

      <section aria-labelledby="other-panels">
        <h2 id="other-panels" className="font-mono text-lg font-semibold">
          Coming soon
        </h2>
        <ul className="mt-2 list-disc pl-5 text-sm text-neutral-600">
          <li>Historical explorer with regime-overlay on price + drawdown</li>
          <li>Method comparison (state-based vs change-point)</li>
          <li>Backtest panel under central + stress cost columns</li>
          <li>Detection-lag explorer per historical crisis</li>
        </ul>
      </section>
    </main>
  );
}
