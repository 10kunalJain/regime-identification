"use client";

import {
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fmtPrice, fmtProb, fmtYearMonth } from "../lib/format";
import { INK, MUTED, PAPER, RULE, stateColor, stateLabel } from "../lib/palette";
import type { RegimePathPoint } from "../lib/types";

interface Span {
  x1: string;
  x2: string;
  state: number;
}

function contiguousSpans(path: RegimePathPoint[]): Span[] {
  const spans: Span[] = [];
  if (path.length === 0) return spans;
  const first = path[0];
  if (!first) return spans;
  let cur = first.state;
  let start = first.data_time;
  let prev = first;
  for (let i = 1; i < path.length; i++) {
    const p = path[i];
    if (!p) continue;
    if (p.state !== cur) {
      spans.push({ x1: start, x2: prev.data_time, state: cur });
      start = p.data_time;
      cur = p.state;
    }
    prev = p;
  }
  spans.push({ x1: start, x2: prev.data_time, state: cur });
  return spans;
}

interface TooltipPayloadEntry {
  value: number;
  payload: RegimePathPoint;
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}

function PathTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const entry = payload[0];
  if (!entry) return null;
  const { state, crisis_prob } = entry.payload;
  return (
    <div className="border border-rule bg-paper px-2 py-1 font-mono text-[10px] tabular leading-tight">
      <div className="text-muted">{label}</div>
      <div>SPY {fmtPrice(entry.value)}</div>
      <div>
        state <span style={{ color: stateColor(state) }}>{stateLabel(state)}</span>
      </div>
      <div>P(crisis) {fmtProb(crisis_prob)}</div>
    </div>
  );
}

export function RegimePath({ path }: { path: RegimePathPoint[] }) {
  const spans = contiguousSpans(path);
  const closes = path.map((p) => p.spy_close).filter((v) => Number.isFinite(v));
  const minClose = closes.length ? Math.min(...closes) : 1;
  const maxClose = closes.length ? Math.max(...closes) : 1;

  return (
    <ResponsiveContainer width="100%" height={360}>
      <ComposedChart data={path} margin={{ top: 16, right: 16, bottom: 4, left: 4 }}>
        <XAxis
          dataKey="data_time"
          tick={{ fontSize: 11, fill: MUTED }}
          tickLine={{ stroke: RULE }}
          axisLine={{ stroke: RULE }}
          minTickGap={72}
          tickFormatter={fmtYearMonth}
        />
        <YAxis
          scale="log"
          domain={[minClose * 0.97, maxClose * 1.03]}
          tick={{ fontSize: 11, fill: MUTED }}
          tickLine={false}
          axisLine={false}
          width={52}
          orientation="right"
          tickFormatter={(v: number) => Math.round(v).toString()}
          allowDataOverflow
        />
        {spans.map((s) => (
          <ReferenceArea
            key={`${s.x1}-${s.state}`}
            x1={s.x1}
            x2={s.x2}
            fill={stateColor(s.state)}
            fillOpacity={s.state === 2 ? 0.22 : s.state === 1 ? 0.1 : 0.06}
            stroke="none"
          />
        ))}
        <Tooltip
          content={<PathTooltip />}
          cursor={{ stroke: RULE, strokeDasharray: "2 3" }}
          wrapperStyle={{ background: PAPER }}
        />
        <Line
          type="monotone"
          dataKey="spy_close"
          stroke={INK}
          strokeWidth={1.2}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
