/**
 * Hero stat. Three of these sit above the fold (state, P(crisis), confidence).
 *
 * The vertical rule between adjacent stats is *not* drawn here — the parent
 * grid uses `divide-x` so it scales when the row is reshaped on mobile.
 */
export function MetricStat({
  label,
  value,
  caption,
  emphasis = false,
  color,
}: {
  label: string;
  value: string;
  caption?: string;
  emphasis?: boolean;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-2 px-6 first:pl-0 last:pr-0">
      <span className="font-mono text-[10px] smallcaps tracking-widest text-muted">
        {label}
      </span>
      <span
        className="font-mono tabular leading-none"
        style={{
          color: color ?? "var(--ink)",
          fontSize: emphasis ? "2.25rem" : "2rem",
          fontWeight: emphasis ? 500 : 400,
        }}
      >
        {value}
      </span>
      {caption ? (
        <span className="font-mono text-[11px] tabular text-muted">{caption}</span>
      ) : null}
    </div>
  );
}
