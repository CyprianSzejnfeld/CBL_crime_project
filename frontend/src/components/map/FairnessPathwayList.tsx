import type { FairnessIndicator } from "../../api/packages";

export function FairnessPathwayList({ indicators }: { indicators?: FairnessIndicator[] }) {
  const rows = (indicators ?? []).filter((item) => item.label || item.value);

  if (!rows.length) {
    return (
      <div className="rounded-lg border border-slate-100 bg-white p-2.5 text-xs text-slate-500">
        No flagged indicator details are available for this cluster.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {rows.map((item, index) => (
        <div key={`${item.label}-${index}`} className="rounded-lg border border-slate-100 bg-white p-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs font-semibold leading-snug text-slate-800">{item.label}</div>
            <span className={badgeClass(item.flagged)}>
              {item.value}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function badgeClass(flagged?: boolean) {
  return flagged === false
    ? "pill shrink-0 bg-slate-100 text-slate-500"
    : "pill shrink-0 bg-red-50 text-red-700";
}
