import { X } from "lucide-react";
import type { WardCriticalnessProps } from "../../api/packages";
import { EmptyState } from "../common";
import { FlagRow, MetricGrid, MiniStat, PlainFact, Section } from "./panelPrimitives";
import { CRITICALNESS_COLORS, fmtInt, fmtNum, fmtPct } from "../../lib/format";

export function WardCriticalnessPanel({
  ward,
  onClose,
}: {
  ward?: WardCriticalnessProps | null;
  onClose: () => void;
}) {
  if (!ward) {
    return <EmptyState title="Select a ward" hint="Click a ward to see its criticalness, active paths and monitor signals." />;
  }
  const paths = splitList(ward.fairness_pathways);
  const traits = splitList(ward.monitor_trait_labels);
  const isMonitorOnly = Number(ward.multipath_count ?? 0) === 0 && Number(ward.monitor_trait_count ?? 0) > 0;

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="flex items-start justify-between gap-2 border-b border-slate-100 p-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{ward.ward_name}</h3>
          <p className="mt-0.5 text-xs text-slate-500">{ward.borough}</p>
          {ward.borough_low_trust && (
            <p className="mt-1 text-xs font-medium text-amber-500">Located in area with low public trust</p>
          )}
        </div>
        <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100"><X className="h-4 w-4" /></button>
      </div>

      <div className="space-y-5 p-4 text-sm">
        <section className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="flex items-center text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            Criticalness
          </div>
          <div className="mt-2 flex items-center gap-2 text-sm font-semibold text-slate-900">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: CRITICALNESS_COLORS[ward.criticalness_level] ?? "#cbd5e1" }} />
            {ward.criticalness_level}
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
            {isMonitorOnly
              ? "Monitor only: this ward has one standalone flag, but not a combined multi-path unfairness flag."
              : "This ward has one or more combined multi-path fairness flags. More paths means stronger review priority."}
          </p>
        </section>

        <Section title="Active paths">
          {paths.length ? paths.map((path) => (
            <PlainFact key={path} label={path} value={pathExplanation(path, ward)} />
          )) : <PlainFact label="No multipath active" value="No combined unfairness path is active. This ward is shown only because it has standalone over-search or very-low-yield evidence." />}
        </Section>

        <Section title="Independent flags">
          <div className="space-y-2">
            {wardFlags(ward).map((flag) => (
              <FlagRow key={flag.label} label={flag.label} active={flag.active} text={flag.text} />
            ))}
          </div>
        </Section>

        {isMonitorOnly ? (
          <Section title="Monitor signals">
            {traits.length ? traits.map((trait) => (
              <PlainFact key={trait} label={trait} value={traitExplanation(trait)} />
            )) : <p className="text-xs text-slate-500">No standalone over-search or very-low-yield signal recorded.</p>}
          </Section>
        ) : null}

        <Section title="Search context">
          <MetricGrid>
            <MiniStat label="Searches / qtr" value={fmtInt(ward.total_stops_qtr)} />
            <MiniStat label="No-result / qtr" value={fmtInt(ward.no_result_stops_qtr)} />
            <MiniStat label="No-result rate" value={fmtPct(ward.no_result_rate)} />
            <MiniStat label="Unfair excess / qtr" value={fmtInt(ward.excess_searches_to_london_normal_qtr)} />
          </MetricGrid>
          <PlainFact
            label="Count searches compared to London avg"
            value={`x${fmtNum(ward.stop_rate_vs_london_avg_ratio, 2)}`}
          />
        </Section>
      </div>
    </div>
  );
}

function splitList(value?: string | null) {
  return String(value ?? "")
    .split(";")
    .map((x) => x.trim())
    .filter(Boolean);
}

function pathExplanation(path: string, ward: WardCriticalnessProps) {
  if (path.includes("Over-search")) return "Search rate is much higher than London average, and yield evidence is very low.";
  if (path.includes("Deprivation")) return "Ward is highly deprived and search rate is above London-normal range.";
  if (path.includes("Racial")) {
    const groups = splitList(ward.racial_oversearch_groups);
    return groups.length
      ? `Group(s): ${groups.join(", ")}. They are over-exposed and their search yield is low against London benchmark.`
      : "A resident group is over-exposed and that group's search yield is low against London benchmark.";
  }
  return "Combined ward-level fairness pathway.";
}

function traitExplanation(trait: string) {
  if (trait.includes("Over-search")) return "Search rate is above London-normal range, but no selected combined path is active.";
  if (trait.includes("Very low yield")) return "Very-low-yield evidence is present, but no selected combined path is active.";
  return "Standalone monitor signal.";
}

function wardFlags(ward: WardCriticalnessProps) {
  return [
    {
      label: "Over-search",
      active: ward.substantial_oversearch_flag,
      text: "Search rate is above London-normal ward range.",
    },
    {
      label: "Much over-search",
      active: ward.much_oversearch_flag,
      text: "Search rate is at least 1.50x London ward average with enough volume.",
    },
    {
      label: "High deprivation",
      active: ward.deprivation_trait_flag,
      text: "Ward sits in high-deprivation range.",
    },
    {
      label: "Low-result category",
      active: Boolean(ward.low_yield_actionability_flag || ward.very_low_yield_actionability_flag),
      text: `Category: ${friendlyCategories(ward.low_yield_categories) || "none"}.`,
    },
    {
      label: "Very-low-result category",
      active: ward.very_low_yield_actionability_flag,
      text: `Category: ${friendlyCategories(ward.very_low_yield_categories) || "none"}.`,
    },
    {
      label: "Racial over-search + low yield",
      active: ward.racial_pathway_flag,
      text: `Group(s): ${splitList(ward.racial_oversearch_groups).join(", ") || "none"}.`,
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
