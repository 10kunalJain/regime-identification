/**
 * Hex palette used by chart libraries that need the raw color value
 * (Recharts, SVG fills). Tailwind utilities pull from these via CSS
 * variables declared in `globals.css`.
 *
 * The palette is intentionally narrow: ink, paper, three state colors,
 * and a single accent that doubles as the crisis color. No gradients.
 */

export const INK = "#1a1614";
export const PAPER = "#faf7f2";
export const PAPER_2 = "#f0ece4";
export const RULE = "#d4cfc4";
export const RULE_2 = "#ebe6dc";
export const MUTED = "#6e6863";

export const STATE_COLORS: readonly string[] = [
  "#5b6470", // 0 — normal expansion (slate)
  "#6b7d5e", // 1 — calm bull / low-vol (sage)
  "#962d22", // 2 — crisis (oxblood)
];

export function stateColor(s: number): string {
  return STATE_COLORS[s] ?? MUTED;
}

export function stateLabel(s: number): string {
  switch (s) {
    case 0:
      return "normal";
    case 1:
      return "calm bull";
    case 2:
      return "crisis";
    default:
      return "unknown";
  }
}

export const CRISIS_THRESHOLD = 0.5;
