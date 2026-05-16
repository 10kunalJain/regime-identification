import type { Freshness } from "../lib/types";

const TONE: Record<Freshness, { label: string; dot: string }> = {
  live: { label: "live", dot: "#1f7a3a" },
  cached: { label: "cached", dot: "#a76a08" },
  static: { label: "static", dot: "var(--muted)" },
};

/**
 * Compact data-freshness chip in the header. Visual: monospace, hairline,
 * single coloured dot. No animation — a pulsing dot would be noise.
 */
export function FreshnessBadge({ kind, asOf }: { kind: Freshness; asOf: string }) {
  const t = TONE[kind];
  return (
    <span className="inline-flex items-center gap-2 border border-rule px-2.5 py-1 font-mono text-[10px] smallcaps tabular tracking-widest text-ink">
      <span
        className="block size-1.5 rounded-full"
        style={{ backgroundColor: t.dot }}
        aria-hidden
      />
      <span>{t.label}</span>
      <span className="text-muted">·</span>
      <span>{asOf}</span>
    </span>
  );
}
