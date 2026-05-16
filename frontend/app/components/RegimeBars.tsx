import { STATE_COLORS, stateLabel } from "../lib/palette";
import { fmtProb } from "../lib/format";

const ROWS: ReadonlyArray<{ key: string; state: number }> = [
  { key: "normal", state: 0 },
  { key: "calm_bull", state: 1 },
  { key: "crisis", state: 2 },
];

/**
 * Horizontal bars for the uncalibrated three-state posterior. Custom HTML —
 * no Recharts — because a 3-row bar chart with right-aligned monospace
 * values is faster, smaller, and crisper than a SVG chart at this size.
 */
export function RegimeBars({ probs }: { probs: Record<string, number> }) {
  return (
    <div className="flex flex-col gap-4">
      {ROWS.map(({ key, state }) => {
        const p = probs[key] ?? 0;
        const width = `${Math.max(0, Math.min(1, p)) * 100}%`;
        return (
          <div key={key} className="flex items-center gap-4">
            <span className="w-24 font-mono text-[11px] tabular smallcaps tracking-wider text-muted">
              {stateLabel(state)}
            </span>
            <div className="relative h-2 flex-1 bg-rule-2">
              <div
                className="absolute inset-y-0 left-0"
                style={{ width, backgroundColor: STATE_COLORS[state] }}
              />
            </div>
            <span className="w-12 text-right font-mono text-sm tabular">{fmtProb(p)}</span>
          </div>
        );
      })}
    </div>
  );
}
