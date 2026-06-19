import { useEffect, useMemo, useState } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";
import { useOptimise, usePackageMap, usePackageScenario, usePackages } from "../api/packages";
import { LoadingSkeleton } from "../components/common";
import { PackageClusterMap } from "../components/map/PackageClusterMap";
import { PackageDetailPanel } from "../components/map/PackageDetailPanel";
import { PACKAGE_COLORS, PACKAGE_LABEL } from "../lib/packages";
import { fmtInt, fmtNum } from "../lib/format";

export function PackageAllocationPage() {
  const { data: packageDefs } = usePackages();
  const [selected, setSelected] = useState<string | null>(null);
  const [budgetScale, setBudgetScale] = useState(() => initialBudgetScale());
  const { data: map, isLoading: mapLoading } = usePackageMap();
  const { data: summary } = usePackageScenario();
  const optimise = useOptimise();

  useEffect(() => {
    localStorage.setItem("packageBudgetScale", String(budgetScale));
    const url = new URL(window.location.href);
    url.searchParams.set("budget_scale", String(budgetScale));
    window.history.replaceState({}, "", url);
    if (Math.abs(budgetScale - 1) < 0.001) return undefined;
    const timer = window.setTimeout(() => {
      optimise.mutate({ budgetScale });
    }, 350);
    return () => window.clearTimeout(timer);
  }, [budgetScale]);

  const optimisedReady = Math.abs(Number(optimise.data?.budget_scale ?? -1) - budgetScale) < 0.001;
  const activeSummary = Math.abs(budgetScale - 1) < 0.001 ? summary : optimisedReady ? optimise.data?.summary : summary;
  const activeMap = Math.abs(budgetScale - 1) < 0.001 ? map : optimisedReady ? optimise.data?.map : map;

  const allocatedFor = useMemo(() => {
    const f = activeMap?.features.find((x) => (x.properties as { cluster_id?: string }).cluster_id === selected);
    return (f?.properties as { allocated_package_id?: string })?.allocated_package_id ?? "P0";
  }, [activeMap, selected]);
  const scaledBudgets = useMemo(() => {
    const base = packageDefs?.quarterly_budgets ?? {};
    return Object.entries(base).map(([key, value]) => ({
      key,
      label: RESOURCE_LABEL[key] ?? key,
      value: Number(value) * budgetScale,
      baseValue: Number(value),
      used: Number(activeSummary?.[`used_${key}`] ?? 0),
    }));
  }, [packageDefs, activeSummary, budgetScale]);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <section className="card space-y-4 p-4">
        <div>
          <div className="mb-2 flex items-center justify-between gap-3">
            <label htmlFor="budget-scale" className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Planning budget
            </label>
            <div className="flex items-center gap-1">
              <IconButton label="Decrease budget" onClick={() => setBudgetScale((value) => clampBudget(value - 0.05))}><Minus className="h-3.5 w-3.5" /></IconButton>
              <span className="min-w-14 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-center text-xs font-semibold text-slate-800">
                x{fmtNum(budgetScale, 2)}
              </span>
              <IconButton label="Increase budget" onClick={() => setBudgetScale((value) => clampBudget(value + 0.05))}><Plus className="h-3.5 w-3.5" /></IconButton>
              <IconButton label="Reset budget" onClick={() => setBudgetScale(1)}><RotateCcw className="h-3.5 w-3.5" /></IconButton>
            </div>
          </div>
          <input
            id="budget-scale"
            type="range"
            min="0.5"
            max="2"
            step="0.05"
            value={budgetScale}
            onChange={(event) => setBudgetScale(clampBudget(Number(event.target.value)))}
            className="w-full accent-brand-600"
          />
        </div>
        <BudgetImpact summary={activeSummary} />
      </section>

      {scaledBudgets.length > 0 ? (
        <section className="card p-3">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Quarterly planning budget</div>
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {scaledBudgets.map((item) => (
              <div key={item.key} className="rounded-lg border border-slate-100 bg-white p-2.5">
                <div className="text-[11px] text-slate-500">{item.label}</div>
                <div className="mt-1 text-sm font-semibold text-slate-900">{budgetValueText(item.key, item.used, item.value)}</div>
                <div className="mt-0.5 text-[11px] leading-relaxed text-slate-400">{budgetHintText(item.key, item.value, item.baseValue, budgetScale)}</div>
                {showBudgetBar(item.key) ? (
                  <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full bg-brand-600"
                      style={{ width: `${Math.min(100, item.value ? (item.used / item.value) * 100 : 0)}%` }}
                    />
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="card overflow-hidden">
        <div className={`grid min-h-[660px] grid-cols-1 ${selected ? "lg:grid-cols-[1fr_400px]" : ""}`}>
          <div className="relative min-h-[540px]">
            {mapLoading && (
              <div className="absolute inset-0 z-10 grid place-items-center bg-white/70"><LoadingSkeleton className="h-40 w-64" /></div>
            )}
            <PackageClusterMap data={activeMap} colorBy="package" selected={selected} onSelect={setSelected} />
            <Legend />
          </div>
          {selected && (
            <aside className="min-h-0 border-t border-slate-200 bg-white lg:border-l lg:border-t-0">
              <PackageDetailPanel clusterId={selected} allocatedPackageId={allocatedFor} onClose={() => setSelected(null)} />
            </aside>
          )}
        </div>
      </section>
    </div>
  );
}

const RESOURCE_LABEL: Record<string, string> = {
  encounter_quality_audits: "Encounter audits",
  procedural_justice_training_places: "Training places",
  community_scrutiny_sessions: "Community scrutiny sessions",
  precision_protection_packages: "Protected-presence packages",
  monitored_search_regime_reviews: "Search-practice review slots",
  evaluation_slots: "Evaluation slots",
};

const SERVICE_RESOURCE_KEYS = new Set(["precision_protection_packages", "evaluation_slots"]);
const CAPACITY_ONLY_RESOURCE_KEYS = new Set(["monitored_search_regime_reviews"]);
const PACKAGE_IDS = ["P1", "P2", "P3", "P4", "P5"];

function BudgetImpact({ summary }: { summary?: Record<string, unknown> }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <StatCard label="Funded units" value={fmtInt(numberField(summary, "clusters_treated"))} />
      <StatCard label="Package mix" value={packageMixText(summary)} />
      <StatCard label="No-result burden covered" value={fmtNum(numberField(summary, "no_result_burden_covered"))} />
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-semibold text-slate-900">{value}</div>
    </div>
  );
}

function numberField(row: Record<string, unknown> | undefined, key: string) {
  const n = Number(row?.[key] ?? 0);
  return Number.isFinite(n) ? n : 0;
}

function packageMixText(row: Record<string, unknown> | undefined) {
  const parts = PACKAGE_IDS.map((id) => {
    const n = numberField(row, `n_${id}`);
    return n > 0 ? `${id} ${fmtInt(n)}` : null;
  }).filter(Boolean);
  return parts.length ? parts.join(", ") : "No funded packages";
}

function showBudgetBar(key: string) {
  return !SERVICE_RESOURCE_KEYS.has(key) && !CAPACITY_ONLY_RESOURCE_KEYS.has(key);
}

function budgetValueText(key: string, used: number, value: number) {
  if (SERVICE_RESOURCE_KEYS.has(key)) return used > 0 ? "Included" : "Not used";
  if (CAPACITY_ONLY_RESOURCE_KEYS.has(key)) return used > 0 ? `${fmtInt(used)} active reviews` : "No active reviews";
  return `${fmtInt(used)} / ${fmtInt(value)}`;
}

function budgetHintText(key: string, value: number, baseValue: number, budgetScale: number) {
  if (key === "precision_protection_packages") return "Service support, not a count target.";
  if (key === "evaluation_slots") return "Evaluation support, not a count target.";
  if (key === "monitored_search_regime_reviews") return `${fmtInt(value)} available; capacity is not the bottleneck.`;
  if (Math.abs(budgetScale - 1) < 0.001) return "Used / available this quarter.";
  return `Base budget ${fmtInt(baseValue)}; selected budget ${fmtInt(value)}.`;
}

function clampBudget(value: number) {
  const next = Math.min(2, Math.max(0.5, value));
  return Number(next.toFixed(2));
}

function initialBudgetScale() {
  const urlValue = Number(new URLSearchParams(window.location.search).get("budget_scale"));
  if (Number.isFinite(urlValue) && urlValue > 0) return clampBudget(urlValue);
  const stored = Number(localStorage.getItem("packageBudgetScale"));
  return Number.isFinite(stored) && stored > 0 ? clampBudget(stored) : 1;
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="grid h-7 w-7 place-items-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900"
    >
      {children}
    </button>
  );
}

function Legend() {
  const items = Object.keys(PACKAGE_LABEL).map((k) => [`${k} · ${PACKAGE_LABEL[k]}`, PACKAGE_COLORS[k]] as const);
  return (
    <div className="absolute bottom-4 left-4 z-10 max-w-[260px] rounded-lg bg-white/95 p-3 text-xs">
      <div className="mb-2 font-semibold text-slate-800">Allocated package</div>
      <div className="space-y-1">
        {items.map(([label, color]) => (
          <div key={label} className="flex items-center gap-2">
            <span className="h-3 w-5 rounded-sm" style={{ background: color }} />
            <span className="text-slate-600">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
