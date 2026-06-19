import type { MapFeatureProps } from "../../types/api";
import { HoverRow as Row } from "./panelPrimitives";

export function HoverTooltip({ p }: { p: MapFeatureProps }) {
  return (
    <div className="w-64 rounded-xl bg-white p-3 text-xs">
      <div className="font-semibold text-slate-900">
        {p.lsoa21nm ?? p.lsoa21cd}
      </div>
      <div className="mt-0.5 text-slate-500">
        {p.lsoa21cd} · {p.borough}
      </div>
      <div className="mt-2 space-y-1 text-slate-600">
        <Row label="Review level">{p.overall_review_priority ?? "No signal"}</Row>
        <Row label="Flagged for">{friendlyPattern(p.dominant_unfairness_pattern)}</Row>
        <Row label="Search pressure">{score(p.excess_burden_score_0_100 ?? p.over_search_score_0_100)}</Row>
        <Row label="Low-result evidence">{score(p.low_yield_actionability_score_0_100 ?? p.low_yield_score_0_100)}</Row>
        <Row label="Monthly crime-risk cap">{p.crime_guardrail_level ?? "Not shown"}</Row>
      </div>
      <div className="mt-2 text-[10px] uppercase tracking-wide text-slate-400">Click for review detail</div>
    </div>
  );
}

function score(value?: number | null) {
  return value === null || value === undefined ? "Not available" : `${Number(value).toFixed(0)} / 100`;
}

function friendlyPattern(value?: string | null) {
  if (!value || value === "No strong unfairness pattern") return "No single strong pattern";
  return value
    .replace(/Deprivation, over-search and average\/lower yield/g, "High search pressure in deprived area")
    .replace(/Racial over-search with average\/lower yield/g, "Group exposure concern")
    .replace(/Extreme over-search with low yield/g, "Very high search pressure, low-result evidence")
    .replace(/_/g, " ");
}
