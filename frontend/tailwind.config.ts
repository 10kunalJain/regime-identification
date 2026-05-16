import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        serif: ["var(--font-serif)", "Georgia", "Times New Roman", "serif"],
        mono: [
          "var(--font-mono)",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      colors: {
        ink: "var(--ink)",
        paper: "var(--paper)",
        "paper-2": "var(--paper-2)",
        rule: "var(--rule)",
        "rule-2": "var(--rule-2)",
        muted: "var(--muted)",
        state: {
          normal: "var(--state-normal)",
          calm: "var(--state-calm)",
          crisis: "var(--state-crisis)",
        },
      },
      letterSpacing: {
        wider: "0.08em",
        widest: "0.14em",
      },
    },
  },
  plugins: [],
};

export default config;
