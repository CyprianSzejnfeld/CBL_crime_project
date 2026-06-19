import { useMemo, useState } from "react";
import { useMapLsoas } from "../api/endpoints";
import { useWardCriticalnessMap } from "../api/packages";
import { LoadingSkeleton, MetricCard } from "../components/common";
import { LondonLsoaMap } from "../components/map/LondonLsoaMap";
import { LsoaDiagnosticPanel } from "../components/map/LsoaDiagnosticPanel";
import { WardCriticalnessMap } from "../components/map/WardCriticalnessMap";
import { WardCriticalnessPanel } from "../components/map/WardCriticalnessPanel";
import { CRITICALNESS_COLORS, fmtInt } from "../lib/format";

type DiagnosticView = "wards" | "lsoas";

export function FairnessV2Page() {
  const { data: wardMap, isLoading } = useWardCriticalnessMap();
  const { data: lsoaMap, isLoading: isLsoaLoading } = useMapLsoas();
  const [view, setView] = useState<DiagnosticView>("wards");
  const [selectedWard, setSelectedWard] = useState<string | null>(null);
  const [selectedLsoa, setSelectedLsoa] = useState<string | null>(null);

  const wardCounts = useMemo(() => {
    const c: Record<string, number> = { "No signal": 0, "Monitor trait": 0, "One multipath": 0, "Two multipaths": 0, "Three multipaths": 0 };
    for (const f of wardMap?.features ?? []) {
      const b = f.properties.criticalness_level ?? "No signal";
      c[b] = (c[b] ?? 0) + 1;
    }
    return c;
  }, [wardMap]);

  const lsoaCounts = useMemo(() => {
    const c: Record<string, number> = { "No signal": 0, "Monitor trait": 0, "One multipath": 0, "Two multipaths": 0, "Three multipaths": 0 };
    for (const f of lsoaMap?.features ?? []) {
      const b = f.properties.criticalness_level ?? "No signal";
      c[b] = (c[b] ?? 0) + 1;
    }
    return c;
  }, [lsoaMap]);

  const showLsoas = view === "lsoas";
  const loading = showLsoas ? isLsoaLoading : isLoading;
  const hasSelection = showLsoas ? Boolean(selectedLsoa) : Boolean(selectedWard);
  const selectedWardProps = useMemo(
    () => wardMap?.features.find((f) => f.properties.ward_code === selectedWard)?.properties ?? null,
    [wardMap, selectedWard],
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-end">
        <div className="inline-flex w-fit rounded-lg border border-slate-200 bg-white p-1 text-xs font-semibold">
          <button
            type="button"
            onClick={() => setView("wards")}
            className={`rounded-md px-3 py-1.5 ${!showLsoas ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-50"}`}
          >
            Wards
          </button>
          <button
            type="button"
            onClick={() => setView("lsoas")}
            className={`rounded-md px-3 py-1.5 ${showLsoas ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-50"}`}
          >
            LSOAs
          </button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        {showLsoas ? (
          <>
            <MetricCard label="LSOAs shown" value={fmtInt(lsoaMap?.features.length)} accent="#334155" />
            <MetricCard label="Monitor flag" value={fmtInt(lsoaCounts["Monitor trait"])} accent={CRITICALNESS_COLORS["Monitor trait"]} />
            <MetricCard label="One multipath" value={fmtInt(lsoaCounts["One multipath"])} accent={CRITICALNESS_COLORS["One multipath"]} />
            <MetricCard label="Two / three multipaths" value={fmtInt(lsoaCounts["Two multipaths"] + lsoaCounts["Three multipaths"])} accent={CRITICALNESS_COLORS["Three multipaths"]} />
          </>
        ) : (
          <>
            <MetricCard label="Wards shown" value={fmtInt(wardMap?.features.length)} accent="#334155" />
            <MetricCard label="Monitor flag" value={fmtInt(wardCounts["Monitor trait"])} accent={CRITICALNESS_COLORS["Monitor trait"]} />
            <MetricCard label="One multipath" value={fmtInt(wardCounts["One multipath"])} accent={CRITICALNESS_COLORS["One multipath"]} />
            <MetricCard label="Two / three multipaths" value={fmtInt(wardCounts["Two multipaths"] + wardCounts["Three multipaths"])} accent={CRITICALNESS_COLORS["Three multipaths"]} />
          </>
        )}
      </div>

      <section className="card overflow-hidden">
        <div className={`grid min-h-[640px] grid-cols-1 ${hasSelection ? "lg:grid-cols-[1fr_400px]" : ""}`}>
          <div className="relative min-h-[520px]">
            {loading && <div className="absolute inset-0 z-10 grid place-items-center bg-white/70"><LoadingSkeleton className="h-40 w-64" /></div>}
            {showLsoas ? (
              <LondonLsoaMap
                data={lsoaMap}
                metric="criticalness_level"
                selected={selectedLsoa}
                onSelect={setSelectedLsoa}
              />
            ) : (
              <WardCriticalnessMap data={wardMap} selected={selectedWard} onSelect={setSelectedWard} />
            )}
            <div className="absolute bottom-3 left-3 z-10 rounded-lg bg-white/95 p-2.5 text-xs">
              <div className="mb-1 font-semibold text-slate-800">Criticalness</div>
              {Object.entries(CRITICALNESS_COLORS).map(([label, color]) => (
                <div key={label} className="flex items-center gap-2"><span className="h-3 w-5 rounded-sm" style={{ background: color }} />{label}</div>
              ))}
            </div>
          </div>
          {hasSelection && (
            <aside className="min-h-0 border-t border-slate-200 bg-white lg:border-l lg:border-t-0">
              {showLsoas ? (
                <LsoaDiagnosticPanel lsoa={selectedLsoa} onClose={() => setSelectedLsoa(null)} />
              ) : (
                <WardCriticalnessPanel ward={selectedWardProps} onClose={() => setSelectedWard(null)} />
              )}
            </aside>
          )}
        </div>
      </section>
    </div>
  );
}
