import { X } from "lucide-react";
import { useClusterDetail, type SearchReviewRow } from "../../api/packages";
import { EmptyState, LoadingSkeleton } from "../common";
import { FairnessPathwayList } from "./FairnessPathwayList";
import { KV, MiniStat, RiskScale, Section } from "./panelPrimitives";
import { PROTECTION_COLORS } from "../../lib/packages";
import { fmtInt, fmtPct } from "../../lib/format";

export function ReductionDetailPanel({
  clusterId,
  reviewRows = [],
  onClose,
}: {
  clusterId: string | null;
  reviewRows?: SearchReviewRow[];
  onClose: () => void;
}) {
  const { data, isLoading } = useClusterDetail(clusterId);

  if (!clusterId) {
    return <EmptyState title="Select a ward" hint="Click the map to review search counts, outcomes and safety level." />;
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-3 p-4">
        <LoadingSkeleton className="h-16" />
        <LoadingSkeleton className="h-32" />
        <LoadingSkeleton className="h-32" />
      </div>
    );
  }

  const rows = reviewRows.length ? reviewRows : data.search_regimes.map((r) => ({
    ...r,
    cluster_id: data.cluster_id,
    cluster_name: data.cluster_name,
    boroughs: data.boroughs,
    is_rollup: r.search_regime === "combined_non_weapon",
    quarterly_no_result_searches: Number(r.quarterly_stops ?? 0) * Number(r.smoothed_no_result_rate ?? 0),
    quarterly_positive_outcomes: Number(r.quarterly_stops ?? 0) * Number(r.smoothed_positive_outcome_rate ?? 0),
    protection_need_band: data.protection?.protection_need_band,
  })) as SearchReviewRow[];

  const atomicRows = rows.filter((r) => !r.is_rollup);
  const sortedRows = [...atomicRows].sort((a, b) => Number(b.quarterly_stops ?? 0) - Number(a.quarterly_stops ?? 0));
  const totalSearches = sum(atomicRows, "quarterly_stops");
  const noResult = sum(atomicRows, "quarterly_no_result_searches");
  const noResultRate = totalSearches ? noResult / totalSearches : 0;
  const targets = atomicRows.filter((r) => r.is_reduction_target);
  const suggestedSearchesReduced = sum(targets, "expected_searches_reduced_if_applied");
  const unfairSearches =
    Number(data.expected_quarterly_unfair_searches_to_london_normal ?? rows[0]?.expected_quarterly_unfair_searches_to_london_normal ?? 0) || 0;
  const excessToLondonAvg =
    Number(data.expected_quarterly_excess_searches_to_london_avg ?? rows[0]?.expected_quarterly_excess_searches_to_london_avg ?? 0) || 0;
  const reductionCap = maxSafetyCap(targets.length ? targets : atomicRows);
  const capLevel = reductionCap <= 0.05 ? "Critical" : reductionCap <= 0.15 ? "High" : reductionCap <= 0.30 ? "Medium" : "Low";
  const capColor = PROTECTION_COLORS[capLevel] ?? "#334155";

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-slate-100 bg-white p-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{friendlyClusterName(data.cluster_name)}</h3>
          <p className="mt-0.5 text-xs text-slate-500">{data.boroughs}</p>
        </div>
        <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="space-y-5 p-4 text-sm">
        <section className={`rounded-lg border p-3 ${targets.length ? "border-blue-200 bg-blue-50" : "border-slate-200 bg-slate-50"}`}>
          <div className={`flex items-center text-[11px] font-semibold uppercase tracking-wide ${targets.length ? "text-blue-700" : "text-slate-500"}`}>
            Suggested action
          </div>
          <div className="mt-1 text-sm font-semibold text-slate-900">
            {targets.length ? targetSummary(targets) : "No search reduction suggested"}
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
            {targets.length
              ? `${targets.length} search type${targets.length === 1 ? " has" : "s have"} an NFA/no-result rate at or above the London average for the same type, enough volume and safety room.`
              :
              "All search types are counted, but none pass the ward fairness, London-average NFA, volume and safety checks for a practical reduction in this ward."}
          </p>
        </section>

        <Section title="Ward totals">
          <div className="grid grid-cols-3 gap-2">
            <MiniStat label="Searches / qtr" value={fmtInt(totalSearches)} />
            <MiniStat label="Unfair excess / qtr" value={fmtInt(unfairSearches)} />
            <MiniStat label="Max cut to avg / qtr" value={fmtInt(excessToLondonAvg)} />
            <MiniStat label="No-result / qtr" value={fmtInt(noResult)} />
            <MiniStat label="No-result rate" value={fmtPct(noResultRate)} />
            <MiniStat label="Suggested cut / qtr" value={targets.length ? fmtInt(suggestedSearchesReduced) : "None"} />
          </div>
        </Section>

        <Section title="Search type breakdown">
          <div className="space-y-2">
            {sortedRows.map((r) => (
              <div key={r.search_regime} className="rounded-lg border border-slate-100 bg-white p-2.5">
                <div className="flex items-center justify-between gap-2 pb-2 border-b border-slate-100">
                  <div className="text-xs font-semibold text-slate-800">{friendlySearch(r.search_regime)}</div>
                  <ActionBadge state={r.recommendation_state} />
                </div>
                <div className="mt-2 grid grid-cols-4 gap-2 text-center">
                  <StatCell label="Cut" value={r.is_reduction_target ? fmtPct(r.recommended_reduction_pct) : "-"} />
                  <StatCell label="Qtr" value={fmtInt(r.quarterly_stops)} />
                  <StatCell label="No result" value={fmtPct(r.smoothed_no_result_rate)} />
                  <StatCell label="Gap" value={fmtPp(r.no_result_rate_gap_vs_london)} highlight={nfaGap(r) >= 0} />
                </div>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Why this ward was flagged">
          <FairnessPathwayList indicators={data.fairness_indicators} />
        </Section>

        <Section title="Safety scale">
          <KV label="Reduction safety cap">
            <span className="font-semibold" style={{ color: capColor }}>
              {fmtPct(reductionCap)}
            </span>
          </KV>
          <KV label="Protection level">{safetyLabel(data.protection?.protection_need_band)}</KV>
          <KV label="Total crime risk">{data.protection?.aggregate_crime_guardrail ?? "Not available"}</KV>
          <RiskScale
            label="Serious-crime burden per 1,000"
            rank={data.protection?.predicted_serious_harm_rank_pct}
            value={data.protection?.predicted_serious_harm_per_1000_residents}
          />
          <RiskScale
            label="Weighted serious harm per 1,000"
            rank={data.protection?.predicted_harm_weighted_serious_crime_score_rank_pct}
            value={data.protection?.predicted_harm_weighted_serious_crime_score_per_1000_residents}
          />
        </Section>
      </div>
    </div>
  );
}

function ActionBadge({ state }: { state?: string | null }) {
  const s = state ?? "No reduction suggested";
  const short = s === "Suggested reduction" ? "Suggested" : s === "Context roll-up" ? "Context" : s === "No reduction suggested" ? "No cut" : s;
  const cls =
    s === "Suggested reduction"
      ? "bg-blue-50 text-blue-700"
      : s === "Review only"
        ? "bg-amber-50 text-amber-700"
        : s === "Blocked by safety"
          ? "bg-sky-50 text-sky-700"
          : "bg-slate-100 text-slate-500";
  return <span className={`pill shrink-0 ${cls}`}>{short}</span>;
}

function StatCell({ label, value, highlight }: { label: string; value: React.ReactNode; highlight?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`text-xs font-semibold ${highlight ? "text-blue-700" : "text-slate-700"}`}>{value}</div>
    </div>
  );
}

function sum(rows: SearchReviewRow[], key: keyof SearchReviewRow) {
  return rows.reduce((total, row) => total + Number(row[key] ?? 0), 0);
}

function maxSafetyCap(rows: SearchReviewRow[]) {
  return rows.reduce((best, row) => Math.max(best, Number(row.safety_reduction_cap ?? 0)), 0);
}

function nfaGap(row: SearchReviewRow) {
  const gap = Number(row.no_result_rate_gap_vs_london);
  if (!Number.isNaN(gap)) return gap;
  const wardRate = Number(row.smoothed_no_result_rate);
  const londonRate = Number(row.london_category_no_result_rate);
  if (Number.isNaN(wardRate) || Number.isNaN(londonRate)) return -Infinity;
  return wardRate - londonRate;
}

function fmtPp(v: number | null | undefined) {
  if (v === null || v === undefined) return "-";
  return `${v >= 0 ? "+" : ""}${(Number(v) * 100).toFixed(0)}pp`;
}

function targetSummary(rows: SearchReviewRow[]) {
  if (rows.length === 1) {
    const r = rows[0];
    return `Reduce ${friendlySearch(r.search_regime)} by ${fmtPct(r.recommended_reduction_pct)}`;
  }
  const types = rows.map((r) => friendlySearch(r.search_regime)).join(", ");
  const pct = Math.max(...rows.map((r) => Number(r.recommended_reduction_pct ?? 0)));
  return `Reduce ${rows.length} selected types up to ${fmtPct(pct)}: ${types}`;
}

function friendlySearch(value?: string | null) {
  const label: Record<string, string> = {
    drugs: "Drug-related",
    stolen_property: "Stolen property",
    other_non_weapon: "Other non-weapon",
    combined_non_weapon: "Combined non-weapon",
    offensive_weapons: "Weapons",
  };
  return value ? label[value] ?? value.replace(/_/g, " ") : "Not available";
}

function friendlyClusterName(value?: string | null) {
  if (!value) return "Selected ward";
  return String(value)
    .replace(/\s+(low_yield_non_weapon|combined_non_weapon|drugs|stolen_property|other_non_weapon|offensive_weapons)\s+ward cluster\s+/g, " ward cluster ")
    .replace(/_/g, " ");
}

function safetyLabel(band?: string | null) {
  const label: Record<string, string> = {
    Low: "Low",
    Medium: "Medium",
    High: "High caution",
    Critical: "Critical",
  };
  return label[String(band ?? "Low")] ?? String(band ?? "Low");
}

