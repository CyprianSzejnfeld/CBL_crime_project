export const fmtPct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "-" : `${(v * 100).toFixed(digits)}%`;
export const fmtNum = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "-" : Number(v).toFixed(digits);
export const fmtInt = (v: number | null | undefined) =>
  v === null || v === undefined ? "-" : Math.round(Number(v)).toLocaleString();

export const CRITICALNESS_COLORS: Record<string, string> = {
  "No signal": "#dbe2ea",
  "Monitor trait": "#fcd34d",
  "One multipath": "#f97316",
  "Two multipaths": "#b91c1c",
  "Three multipaths": "#b91c1c",
};


export type MetricId = "criticalness_level";

export function fillColorExpression(metric: MetricId): unknown {
  return [
    "match",
    ["coalesce", ["get", metric], "No signal"],
    "Monitor trait", CRITICALNESS_COLORS["Monitor trait"],
    "One multipath", CRITICALNESS_COLORS["One multipath"],
    "Two multipaths", CRITICALNESS_COLORS["Two multipaths"],
    "Three multipaths", CRITICALNESS_COLORS["Three multipaths"],
    CRITICALNESS_COLORS["No signal"],
  ];
}
