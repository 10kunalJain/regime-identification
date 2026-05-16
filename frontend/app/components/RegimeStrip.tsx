import { stateColor, stateLabel } from "../lib/palette";
import type { RegimePathPoint } from "../lib/types";

/**
 * Signature element: a horizontal density strip of the last 365 trading
 * days, one pixel-column per day, coloured by the joint-HMM-inferred
 * regime state. Hover reveals the date + state. Tufte-style ambient
 * data display — visible from across a room.
 */
export function RegimeStrip({ path }: { path: RegimePathPoint[] }) {
  if (path.length === 0) return null;
  const first = path[0];
  const last = path[path.length - 1];
  if (!first || !last) return null;

  return (
    <div className="border-y border-rule bg-paper-2">
      <div className="flex h-3 w-full">
        {path.map((p) => (
          <div
            key={p.data_time}
            className="flex-1 transition-opacity hover:opacity-50"
            style={{ backgroundColor: stateColor(p.state) }}
            title={`${p.data_time} · ${stateLabel(p.state)}`}
          />
        ))}
      </div>
      <div className="flex items-center justify-between px-6 py-1.5 font-mono text-[10px] tabular text-muted">
        <span>{first.data_time}</span>
        <span className="smallcaps tracking-widest">last 365 sessions · regime state</span>
        <span>{last.data_time}</span>
      </div>
    </div>
  );
}
