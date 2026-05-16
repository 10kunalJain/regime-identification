import latestStatic from "../public/regime_latest.json";
import pathStatic from "../public/regime_path.json";

import { CrisisSparkline } from "./components/CrisisSparkline";
import { FreshnessBadge } from "./components/FreshnessBadge";
import { MetricStat } from "./components/MetricStat";
import { RegimeBars } from "./components/RegimeBars";
import { RegimePath } from "./components/RegimePath";
import { RegimeStrip } from "./components/RegimeStrip";
import { resolveLatest } from "./lib/api";
import { fmtProb } from "./lib/format";
import { STATE_COLORS, stateLabel } from "./lib/palette";
import type { RegimePathPoint, RegimePosterior } from "./lib/types";

// Vercel ISR — 60s revalidation per ARCHITECTURE.md §10. When the FastAPI
// backend is reachable we stream a fresh posterior; otherwise the static
// JSON baked into /public renders without a network call.
export const revalidate = 60;

const STATE_KEYS: ReadonlyArray<{ key: string; state: number }> = [
  { key: "normal", state: 0 },
  { key: "calm_bull", state: 1 },
  { key: "crisis", state: 2 },
];

function dominantState(probs: Record<string, number>): { key: string; state: number; p: number } {
  let best = { key: "normal", state: 0, p: -Infinity };
  for (const { key, state } of STATE_KEYS) {
    const p = probs[key] ?? 0;
    if (p > best.p) best = { key, state, p };
  }
  if (best.p === -Infinity) best = { key: "normal", state: 0, p: 0 };
  return best;
}

export default async function HomePage() {
  const { posterior, freshness } = await resolveLatest(latestStatic as RegimePosterior);
  const path = pathStatic as RegimePathPoint[];
  const dom = dominantState(posterior.regime_probs_uncal);
  const crisisProb = posterior.crisis_prob_21d_cal;
  const crisisColor = STATE_COLORS[2] ?? "var(--state-crisis)";
  const inCrisis = dom.state === 2 || crisisProb > 0.5;

  return (
    <main className="min-h-screen">
      <Header asOf={posterior.as_of} freshness={freshness} version={posterior.version} />
      <RegimeStrip path={path} />

      <section className="mx-auto max-w-7xl px-6 pb-16 pt-12">
        <Intro />

        <div className="mt-10 grid grid-cols-3 divide-x divide-rule border-y border-rule py-8">
          <MetricStat
            label="state"
            value={stateLabel(dom.state)}
            caption={`P=${fmtProb(dom.p)} · ${posterior.method}`}
            emphasis
            color={dom.state === 2 ? crisisColor : undefined}
          />
          <MetricStat
            label="P(crisis, 21d)"
            value={fmtProb(crisisProb)}
            caption={inCrisis ? "above 0.5 threshold" : "below 0.5 threshold"}
            color={crisisProb > 0.5 ? crisisColor : undefined}
          />
          <MetricStat
            label="confidence"
            value={fmtProb(posterior.confidence)}
            caption="entropy-derived"
          />
        </div>

        <Section
          title="SPY + regime path"
          tail="365 sessions · log price · band by argmax state"
          className="mt-14"
        >
          <RegimePath path={path} />
        </Section>

        <div className="mt-14 grid gap-10 md:grid-cols-2">
          <Section title="Posterior" tail="3-state · uncalibrated">
            <div className="px-1 py-2">
              <RegimeBars probs={posterior.regime_probs_uncal} />
            </div>
          </Section>
          <Section title="P(crisis, 21d)" tail="365 sessions · calibrated · 0.5 anchor">
            <CrisisSparkline path={path} />
          </Section>
        </div>

        <ComingSoon />
      </section>

      <Footer method={posterior.method} version={posterior.version} />
    </main>
  );
}

function Header({
  asOf,
  freshness,
  version,
}: {
  asOf: string;
  freshness: "live" | "cached" | "static";
  version: string;
}) {
  return (
    <header className="border-b border-rule">
      <div className="mx-auto flex max-w-7xl items-baseline justify-between gap-6 px-6 py-5">
        <div className="flex items-baseline gap-3">
          <span className="font-serif text-[15px] smallcaps tracking-widest">Regime</span>
          <span className="font-mono text-xs text-muted">//</span>
          <span className="font-serif text-[15px] smallcaps tracking-widest">
            Identification
          </span>
          <span className="ml-3 hidden font-mono text-[10px] tabular text-muted md:inline">
            v{version}
          </span>
        </div>
        <FreshnessBadge kind={freshness} asOf={asOf} />
      </div>
    </header>
  );
}

function Intro() {
  return (
    <div className="grid gap-6 md:grid-cols-[1.4fr_1fr] md:items-end">
      <h1 className="font-serif text-3xl font-light leading-[1.15] tracking-tight text-ink md:text-[2.5rem]">
        A <em className="italic">filtered-only</em> fair-evaluation benchmark
        <br />
        for US-equity regime identification.
      </h1>
      <p className="max-w-prose font-serif text-[15px] leading-relaxed text-muted">
        A three-state joint cross-sectional HMM with Fama-French regime-switching
        means and rank-3 latent factor covariance, trained on 23 years of SPY,
        sector SPDR, and style-factor returns. No smoothing. No look-ahead.
      </p>
    </div>
  );
}

function Section({
  title,
  tail,
  children,
  className,
}: {
  title: string;
  tail: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={className}>
      <header className="mb-4 flex items-baseline justify-between border-b border-rule-2 pb-2">
        <h2 className="font-serif text-base smallcaps tracking-widest text-ink">{title}</h2>
        <span className="font-mono text-[10px] smallcaps tabular tracking-widest text-muted">
          {tail}
        </span>
      </header>
      <div className="border border-rule bg-paper p-4 md:p-6">{children}</div>
    </section>
  );
}

function ComingSoon() {
  const panels = [
    { name: "historical explorer", note: "range overlay + drawdown" },
    { name: "method comparison", note: "state vs change-point" },
    { name: "backtest panel", note: "central · stress cost columns" },
    { name: "detection-lag explorer", note: "anchor sensitivity, per crisis" },
  ];
  return (
    <section className="mt-16">
      <header className="mb-4 flex items-baseline justify-between border-b border-rule-2 pb-2">
        <h2 className="font-serif text-base smallcaps tracking-widest text-ink">
          Forthcoming
        </h2>
        <span className="font-mono text-[10px] smallcaps tabular tracking-widest text-muted">
          weeks 11 · 12
        </span>
      </header>
      <ul className="grid gap-x-10 gap-y-3 md:grid-cols-2">
        {panels.map((p) => (
          <li
            key={p.name}
            className="flex items-baseline justify-between border-b border-rule-2 py-1 font-mono text-[12px] tabular text-muted"
          >
            <span className="smallcaps tracking-wider">{p.name}</span>
            <span className="text-[11px]">{p.note}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Footer({ method, version }: { method: string; version: string }) {
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto flex max-w-7xl flex-col items-start gap-3 px-6 py-6 text-[12px] md:flex-row md:items-center md:justify-between">
        <span className="font-serif italic text-muted">
          filtered-only fair-evaluation benchmark · joint HMM · k=3
        </span>
        <nav className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono tabular text-muted smallcaps tracking-widest">
          <span aria-disabled="true" className="cursor-not-allowed">
            paper · forthcoming
          </span>
          <a
            className="transition-colors hover:text-ink"
            href="https://github.com/10kunalJain/regime-identification"
            rel="noopener noreferrer"
            target="_blank"
          >
            github
          </a>
          <span>
            {method} · {version}
          </span>
        </nav>
      </div>
    </footer>
  );
}
