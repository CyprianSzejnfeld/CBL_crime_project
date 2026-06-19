import { X } from "lucide-react";
import { useClusterDetail, type ClusterPackageRow } from "../../api/packages";
import { EmptyState, LoadingSkeleton } from "../common";
import { FairnessPathwayList } from "./FairnessPathwayList";
import { KV, PlainFact, RiskScale, Section } from "./panelPrimitives";
import { PACKAGE_COLORS, PACKAGE_EXPLAINER, PACKAGE_LABEL, PROTECTION_COLORS } from "../../lib/packages";
import { fmtInt } from "../../lib/format";

const RESOURCE_LABEL: Record<string, string> = {
  cost_encounter_quality_audits: "Encounter audits",
  cost_procedural_justice_training_places: "Training places",
  cost_community_scrutiny_sessions: "Community scrutiny sessions",
  cost_precision_protection_packages: "Protected-presence packages",
  cost_monitored_search_regime_reviews: "Search-practice review slots",
  cost_evaluation_slots: "Evaluation slots",
};

const SERVICE_RESOURCE_KEYS = new Set(["cost_precision_protection_packages", "cost_evaluation_slots"]);

export function PackageDetailPanel({
  clusterId,
  allocatedPackageId,
  onClose,
}: {
  clusterId: string | null;
  allocatedPackageId: string;
  onClose: () => void;
}) {
  const { data, isLoading } = useClusterDetail(clusterId);

  if (!clusterId) return <EmptyState title="Select a ward" hint="Click a ward on the map for its package, reason, resources and safety level." />;
  if (isLoading || !data) return <div className="p-4"><LoadingSkeleton className="h-72" /></div>;

  const allocated = data.packages.find((p) => p.package_id === allocatedPackageId);
  const packageCopy = PACKAGE_EXPLAINER[allocatedPackageId] ?? PACKAGE_EXPLAINER.P0;

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="flex items-start justify-between gap-2 border-b border-slate-100 p-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{friendlyClusterName(data.cluster_name)}</h3>
          <p className="mt-0.5 text-xs text-slate-500">{data.boroughs}</p>
        </div>
        <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100"><X className="h-4 w-4" /></button>
      </div>

      <div className="space-y-5 p-4 text-sm">
        <section className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="flex items-center text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            Package decision
          </div>
          <div className="mt-2 flex items-center gap-2 text-sm font-semibold text-slate-900">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: PACKAGE_COLORS[allocatedPackageId] }} />
            {PACKAGE_LABEL[allocatedPackageId] ?? allocatedPackageId}
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
            {packageCopy.summary}
          </p>
        </section>

        <Section dense title="Key facts">
          <PlainFact label="When this package is used" value={packageCopy.when} />
          <PlainFact label="What gets funded here" value={packageResourceSentence(allocatedPackageId, allocated)} />
        </Section>

        <Section dense title="Why this ward was flagged">
          <FairnessPathwayList indicators={data.fairness_indicators} />
        </Section>

        <Section dense title="Safety level">
          {data.protection ? (
            <>
              <KV label="Combined safety level">
                <span className="font-semibold" style={{ color: PROTECTION_COLORS[data.protection.protection_need_band] ?? "#334155" }}>
                  {safetyLabel(data.protection.protection_need_band)}
                </span>
              </KV>
              <KV label="Total crime risk">{data.protection.aggregate_crime_guardrail}</KV>
              <RiskScale
                label="Serious-crime burden per 1,000"
                rank={data.protection.predicted_serious_harm_rank_pct}
                value={data.protection.predicted_serious_harm_per_1000_residents}
              />
              <RiskScale
                label="Weighted serious harm per 1,000"
                rank={data.protection.predicted_harm_weighted_serious_crime_score_rank_pct}
                value={data.protection.predicted_harm_weighted_serious_crime_score_per_1000_residents}
              />
            </>
          ) : <p className="text-xs text-slate-400">No forecast.</p>}
        </Section>
      </div>
    </div>
  );
}

function packageResourceSentence(packageId: string, row?: ClusterPackageRow) {
  if (!row || packageId === "P0") return "Monitoring package.";
  const items = Object.keys(RESOURCE_LABEL)
    .map((k) => ({ key: k, label: RESOURCE_LABEL[k], value: Number(row[k] ?? 0) }))
    .filter((i) => i.value > 0)
    .map((i) => resourcePhrase(i.key, i.label, i.value));
  return items.length ? items.join(", ") + "." : "No extra package resource.";
}

function resourcePhrase(key: string, label: string, value: number) {
  const service: Record<string, string> = {
    cost_precision_protection_packages: "protected-presence support",
    cost_evaluation_slots: "evaluation support",
  };
  if (SERVICE_RESOURCE_KEYS.has(key)) return service[key] ?? label.toLowerCase();
  const singular: Record<string, string> = {
    "Encounter audits": "encounter audit",
    "Training places": "training place",
    "Community scrutiny sessions": "community scrutiny session",
    "Protected-presence packages": "protected-presence package",
    "Search-practice review slots": "search-practice review slot",
    "Evaluation slots": "evaluation slot",
  };
  const name = value === 1 ? singular[label] ?? label.toLowerCase() : label.toLowerCase();
  return value === 1 ? name : `${fmtInt(value)} ${name}`;
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
