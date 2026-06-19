import { useMemo, useState } from "react";
import { useSearchReview, useSearchReviewMap } from "../api/packages";
import { LoadingSkeleton, MetricCard } from "../components/common";
import { PackageClusterMap } from "../components/map/PackageClusterMap";
import { ReductionDetailPanel } from "../components/map/ReductionDetailPanel";
import { fmtInt } from "../lib/format";

export function ReductionAnalyticsPage() {
  const { data, isLoading } = useSearchReview();
  const { data: map } = useSearchReviewMap();
  const [selected, setSelected] = useState<string | null>(null);

  const selectedRows = useMemo(
    () => (data?.rows ?? []).filter((r) => r.cluster_id === selected),
    [data, selected],
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {isLoading || !data ? (
        <LoadingSkeleton className="h-72" />
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            <MetricCard
              label="Wards with suggested cut"
              value={fmtInt(data.summary.clusters_with_suggested_reduction)}
              sub={`${fmtInt(data.summary.clusters)} wards counted`}
              accent="#2563eb"
            />
            <MetricCard
              label="Unfair excess / qtr"
              value={fmtInt(data.summary.total_unfair_searches_detected ?? 0)}
              sub="Above London-normal ward rate"
              accent="#dc2626"
            />
            <MetricCard
              label="Suggested searches cut / qtr"
              value={fmtInt(data.summary.total_suggested_searches_reduced)}
              sub="Safe type-specific action"
            />
            <MetricCard
              label="No-result avoided / qtr"
              value={fmtInt(data.summary.total_suggested_no_result_avoided)}
              accent="#b45309"
            />
            <MetricCard
              label="Positive outcomes at risk"
              value={fmtInt(data.summary.total_positive_outcomes_at_risk)}
              sub="Trade-off if applied"
            />
          </div>

          <section className="card overflow-hidden">
            <div className={`grid min-h-[620px] grid-cols-1 ${selected ? "lg:grid-cols-[1fr_390px]" : ""}`}>
              <div className="relative min-h-[480px]">
                <PackageClusterMap data={map} colorBy="search" selected={selected} onSelect={setSelected} />
                <div className="absolute bottom-3 left-3 z-10 rounded-lg bg-white/95 p-2.5 text-xs">
                  <div className="mb-1 font-semibold text-slate-800">Suggested cut</div>
                  {[
                    ["None", "#e2e8f0"],
                    ["<15%", "#bfdbfe"],
                    ["15-20%", "#ef4444"],
                    ["20-30%", "#92400e"],
                    ["≥30%", "#111827"],
                  ].map(([l, c]) => (
                    <div key={l} className="flex items-center gap-2"><span className="h-3 w-5 rounded-sm" style={{ background: c }} />{l}</div>
                  ))}
                </div>
              </div>
              {selected && (
                <aside className="min-h-0 border-t border-slate-200 bg-white lg:border-l lg:border-t-0">
                  <ReductionDetailPanel clusterId={selected} reviewRows={selectedRows} onClose={() => setSelected(null)} />
                </aside>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
