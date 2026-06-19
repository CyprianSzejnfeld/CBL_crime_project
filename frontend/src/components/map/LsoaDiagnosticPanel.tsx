import { X } from "lucide-react";
import { useLsoaDetail } from "../../api/endpoints";
import { EmptyState, LoadingSkeleton } from "../common";
import { FlagRow, MetricGrid, MiniStat, PlainFact, Section } from "./panelPrimitives";
import { CRITICALNESS_COLORS, fmtNum } from "../../lib/format";

export function LsoaDiagnosticPanel({
  lsoa,
  onClose,
}: {
  lsoa: string | null;
  onClose: () => void;
}) {
  const { data, isLoading } = useLsoaDetail(lsoa);

  if (!lsoa) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <EmptyState title="Select an LSOA" hint="Click a neighbourhood to see why it is marked for fairness review." />
      </div>
    );
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-3 p-4">
        <LoadingSkeleton className="h-16" />
        <LoadingSkeleton className="h-40" />
        <LoadingSkeleton className="h-32" />
      </div>
    );
  }

  const paths = activePaths(data);
  const flags = independentFlags(data);
  const pathCount = paths.length;
  const flaggedIndicators = flags.filter((flag) => Boolean(flag.active));

  const monitorIndicators = flaggedIndicators.filter((flag) => flag.label !== "High deprivation");
  const isMonitorOnly = pathCount === 0 && monitorIndicators.length > 0;
  const criticalness = resolvedCriticalness(pathCount, monitorIndicators.length);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="flex items-start justify-between gap-2 border-b border-slate-100 p-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{data.lsoa21nm ?? data.lsoa21cd}</h3>
          <p className="mt-0.5 text-xs text-slate-500">
            {data.lsoa21cd} · {data.borough}
          </p>
          {data.borough_low_trust_flag && (
            <p className="mt-1 text-xs font-medium text-amber-500">Located in area with low public trust</p>
          )}
        </div>
        <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="space-y-5 p-4 text-sm">
        <section className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Criticalness</div>
          <div className="mt-2 flex items-center gap-2 text-sm font-semibold text-slate-900">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: CRITICALNESS_COLORS[criticalness] ?? "#cbd5e1" }} />
            {criticalness}
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
            {isMonitorOnly
              ? "Monitor only: this LSOA has standalone flags, but not a combined multi-path unfairness flag."
              : pathCount > 0
              ? "This LSOA has one or more combined multi-path fairness flags. More paths means stronger review priority."
              : "No selected path or standalone indicator is flagged for this LSOA."}
          </p>
        </section>

        <Section dense title="Active paths">
          {paths.length ? paths.map((path) => (
            <PlainFact key={path.title} label={path.title} value={path.text} />
          )) : <PlainFact label="No multipath active" value={isMonitorOnly ? "No combined unfairness path is active. This LSOA is shown only because it has standalone flagged indicators." : "No combined unfairness path is active, and no standalone indicator is flagged."} />}
        </Section>

        <Section dense title="Independent flags">
          <div className="space-y-2">
            {flags.map((flag) => (
              <FlagRow key={flag.label} label={flag.label} active={flag.active} text={flag.text} />
            ))}
          </div>
        </Section>

        {isMonitorOnly ? (
          <Section dense title="Monitor signals">
            {monitorIndicators.map((flag) => (
              <PlainFact key={flag.label} label={flag.label} value={traitExplanation(flag.label)} />
            ))}
          </Section>
        ) : null}

        <Section dense title="Search context">
          <MetricGrid>
            <MiniStat label="Rate / 1k" value={fmtMaybe(data.stop_rate_per_1000)} />
            <MiniStat label="London avg / 1k" value={fmtMaybe(data.london_avg_lsoa_stop_rate_per_1000)} />
            <MiniStat label="London-normal / 1k" value={fmtMaybe(data.london_normal_lsoa_stop_rate_per_1000)} />
            <MiniStat label="Excess / month" value={fmtMaybe(data.excess_searches_to_london_lsoa_normal_month, 1)} />
          </MetricGrid>
          <PlainFact
            label="Count searches compared to London avg"
            value={`x${fmtMaybe(data.stop_rate_vs_london_lsoa_avg_ratio, 2)}`}
          />
          {data.resident_denominator_caution_flag ? (
            <PlainFact
              label="Resident denominator caution"
              value="Central or high-footfall area. Resident population is a weak comparison point, so read resident-rate disproportionality with care."
            />
          ) : null}
        </Section>
      </div>
    </div>
  );
}

function resolvedCriticalness(pathCount: number, flagCount: number) {
  if (pathCount >= 3) return "Three multipaths";
  if (pathCount === 2) return "Two multipaths";
  if (pathCount === 1) return "One multipath";
  if (flagCount > 0) return "Monitor trait";
  return "No signal";
}

type SignalSource = {
  stop_rate_per_1000?: number | null;
  london_avg_lsoa_stop_rate_per_1000?: number | null;
  london_normal_lsoa_stop_rate_per_1000?: number | null;
  stop_rate_vs_london_lsoa_avg_ratio?: number | null;
  excess_searches_to_london_lsoa_normal_annual?: number | null;
  excess_burden_flag?: boolean | null;
  substantial_oversearch_flag?: boolean | null;
  much_oversearch_flag?: boolean | null;
  oversearch_trait_flag?: boolean | null;
  much_oversearch_trait_flag?: boolean | null;
  deprivation_trait_flag?: boolean | null;
  deprivation_burden_flag?: boolean | null;
  racial_trait_flag?: boolean | null;
  racial_pathway_black_flag?: boolean | null;
  racial_pathway_asian_flag?: boolean | null;
  racial_pathway_mixed_flag?: boolean | null;
  racial_pathway_other_flag?: boolean | null;
  racial_pathway_white_flag?: boolean | null;
  low_yield_actionability_flag?: boolean | null;
  very_low_yield_actionability_flag?: boolean | null;
  deprivation_oversearch_low_yield_flag?: boolean | null;
  deprivation_oversearch_path_flag?: boolean | null;
  racial_oversearch_low_yield_flag?: boolean | null;
  racial_oversearch_low_yield_path_flag?: boolean | null;
  extreme_oversearch_low_yield_flag?: boolean | null;
  oversearch_low_yield_path_flag?: boolean | null;
  eligible_reduction_categories?: string | null;
  very_low_yield_categories?: string | null;
  monitor_trait_labels?: string | null;
};

function activePaths(d: SignalSource) {
  const rows: { title: string; text: string }[] = [];
  if (d.oversearch_low_yield_path_flag || d.extreme_oversearch_low_yield_flag) {
    rows.push({
      title: "Over-search + very low yield",
      text: "Search rate is much above London LSOA average and at least one search type has very-low-result evidence.",
    });
  }
  if (d.deprivation_burden_flag || d.deprivation_oversearch_path_flag || d.deprivation_oversearch_low_yield_flag) {
    rows.push({
      title: "Deprivation burden",
      text: "Area is highly deprived and search rate is above the London-normal LSOA range.",
    });
  }
  if (
    d.racial_oversearch_low_yield_path_flag ||
    d.racial_oversearch_low_yield_flag
  ) {
    rows.push({
      title: "Racial over-search + low yield",
      text: `Group(s): ${racialGroups(d).join(", ") || "recorded group"}. Over-exposed relative to resident share and low yield against London LSOA benchmark.`,
    });
  }
  return rows;
}

function independentFlags(d: SignalSource) {
  const overSearch = Boolean(d.oversearch_trait_flag || d.substantial_oversearch_flag || d.excess_burden_flag || aboveNormalSearch(d));
  const muchOverSearch = Boolean(d.much_oversearch_trait_flag || d.much_oversearch_flag || muchAboveAverageSearch(d));
  const racial = Boolean(d.racial_oversearch_low_yield_path_flag || d.racial_oversearch_low_yield_flag);
  return [
    {
      label: "Over-search",
      active: overSearch,
      text: "LSOA search rate is above London-normal range.",
    },
    {
      label: "Much over-search",
      active: muchOverSearch,
      text: "LSOA search rate is at least 1.50x London LSOA average.",
    },
    {
      label: "High deprivation",
      active: d.deprivation_trait_flag,
      text: "LSOA sits in high-deprivation range.",
    },
    {
      label: "Low-result search type",
      active: Boolean(d.low_yield_actionability_flag || d.very_low_yield_actionability_flag),
      text: `Category evidence: ${friendlyCategories(d.eligible_reduction_categories) || "none flagged"}.`,
    },
    {
      label: "Very-low-result search type",
      active: d.very_low_yield_actionability_flag,
      text: `Very-low-yield category: ${friendlyCategories(d.very_low_yield_categories) || "none flagged"}.`,
    },
    {
      label: "Racial over-search + low yield",
      active: racial,
      text: `Group(s): ${racialGroups(d).join(", ") || "racial signal present"}.`,
    },
  ];
}

function friendlyCategories(value?: string | null) {
  const labels: Record<string, string> = {
    drugs: "Drug-related",
    stolen_property: "Stolen property",
    other_non_weapon: "Other non-weapon",
    offensive_weapons: "Weapons",
    low_yield_non_weapon: "Low-result non-weapon",
  };
  return splitList(value).map((x) => labels[x] ?? x.replace(/_/g, " ")).join(", ");
}

function racialGroups(d: SignalSource) {
  const groups = [];
  if (d.racial_pathway_black_flag) groups.push("Black");
  if (d.racial_pathway_asian_flag) groups.push("Asian");
  if (d.racial_pathway_mixed_flag) groups.push("Mixed");
  if (d.racial_pathway_other_flag) groups.push("Other");
  if (d.racial_pathway_white_flag) groups.push("White");
  return groups;
}

function aboveNormalSearch(d: SignalSource) {
  const rate = Number(d.stop_rate_per_1000);
  const normal = Number(d.london_normal_lsoa_stop_rate_per_1000);
  if (!Number.isFinite(rate) || !Number.isFinite(normal)) return false;
  return rate > normal && Number(d.excess_searches_to_london_lsoa_normal_annual ?? 1) > 0;
}

function muchAboveAverageSearch(d: SignalSource) {
  const ratio = Number(d.stop_rate_vs_london_lsoa_avg_ratio);
  if (!Number.isFinite(ratio)) return false;
  return ratio >= 1.5 && aboveNormalSearch(d);
}

function splitList(value?: string | null) {
  return String(value ?? "")
    .split(";")
    .map((x) => x.trim())
    .filter(Boolean);
}

function traitExplanation(trait: string) {
  const t = trait.toLowerCase();
  if (t.includes("over-search")) return "Search rate is above London-normal range, but no selected combined path is active.";
  if (t.includes("yield") || t.includes("result")) return "Yield evidence is flagged, but no selected combined path is active.";
  if (t.includes("deprivation")) return "High deprivation is present, but no selected combined path is active.";
  if (t.includes("racial")) return "Racial signal is present, but no selected combined path is active.";
  return "Standalone monitor signal.";
}

function fmtMaybe(value?: number | null, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "n/a";
  return fmtNum(Number(value), digits);
}
