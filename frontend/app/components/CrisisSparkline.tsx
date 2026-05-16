"use client";

import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fmtProb, fmtYearMonth } from "../lib/format";
import { CRISIS_THRESHOLD, MUTED, PAPER, RULE, STATE_COLORS } from "../lib/palette";
import type { RegimePathPoint } from "../lib/types";

interface TooltipPayloadEntry {
  value: number;
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}

function SparkTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const entry = payload[0];
  if (!entry) return null;
  return (
    <div
      className="border border-rule bg-paper px-2 py-1 font-mono text-[10px] tabular"
      style={{ boxShadow: "none" }}
    >
      <div className="text-muted">{label}</div>
      <div>P(crisis) {fmtProb(entry.value)}</div>
    </div>
  );
}

export function CrisisSparkline({ path }: { path: RegimePathPoint[] }) {
  const crisisColor = STATE_COLORS[2] ?? MUTED;
  return (
    <ResponsiveContainer width="100%" height={170}>
      <LineChart data={path} margin={{ top: 12, right: 16, bottom: 4, left: 4 }}>
        <XAxis
          dataKey="data_time"
          tick={{ fontSize: 10, fill: MUTED }}
          tickLine={{ stroke: RULE }}
          axisLine={{ stroke: RULE }}
          minTickGap={56}
          tickFormatter={fmtYearMonth}
        />
        <YAxis
          domain={[0, 1]}
          ticks={[0, CRISIS_THRESHOLD, 1]}
          tick={{ fontSize: 10, fill: MUTED }}
          axisLine={false}
          tickLine={false}
          width={28}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <ReferenceLine y={CRISIS_THRESHOLD} stroke={RULE} strokeDasharray="2 3" />
        <Tooltip
          content={<SparkTooltip />}
          cursor={{ stroke: RULE, strokeDasharray: "2 3" }}
          wrapperStyle={{ background: PAPER }}
        />
        <Line
          type="monotone"
          dataKey="crisis_prob"
          stroke={crisisColor}
          strokeWidth={1.1}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
