/** Number + date formatters. Mono numbers use `tabular-nums` via CSS. */

export function fmtProb(n: number): string {
  return n.toFixed(3);
}

export function fmtPct(n: number, decimals = 1): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

export function fmtPrice(n: number): string {
  return n.toFixed(2);
}

export function fmtDateShort(iso: string): string {
  return iso.slice(0, 10);
}

export function fmtYearMonth(iso: string): string {
  return iso.slice(0, 7);
}
