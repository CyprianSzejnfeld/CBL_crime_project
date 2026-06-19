import { fmtNum } from "../../lib/format";

export function Section({ title, children, dense }: { title: string; children: React.ReactNode; dense?: boolean }) {
  return (
    <section>
      <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h4>
      <div className={dense ? "space-y-1" : "space-y-2"}>{children}</div>
    </section>
  );
}

export function KV({ label, children }: { label: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3 text-xs">
      <span className="text-slate-400">{label}</span>
      <span className="text-right font-medium text-slate-700">{children}</span>
    </div>
  );
}

export function PlainFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-100 bg-white p-2.5">
      <div className="text-[11px] font-medium text-slate-500">{label}</div>
      <p className="mt-0.5 text-xs font-semibold leading-relaxed text-slate-800">{value}</p>
    </div>
  );
}

export function MetricGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-2">{children}</div>;
}

export function MiniStat({ label, value }: { label: React.ReactNode; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-100 bg-white p-2.5">
      <div className="text-[10px] font-medium uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-sm font-semibold text-slate-900">{value}</div>
    </div>
  );
}

export function FlagRow({ label, active, text }: { label: string; active?: boolean | null; text: string }) {
  return (
    <div className="rounded-lg border border-slate-100 bg-white p-2.5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-medium text-slate-500">{label}</div>
          <p className="mt-0.5 text-xs font-semibold leading-relaxed text-slate-800">{text}</p>
        </div>
        <span className={`pill shrink-0 ${active ? "bg-red-50 text-red-700" : "bg-slate-100 text-slate-500"}`}>
          {active ? "flagged" : "clear"}
        </span>
      </div>
    </div>
  );
}

export function RiskScale({
  label,
  rank,
  value,
}: {
  label: string;
  rank?: number | null;
  value?: number | null;
}) {
  const pct = rank === undefined || rank === null || Number.isNaN(Number(rank)) ? null : Math.max(0, Math.min(100, Number(rank) * 100));
  const rate = value === undefined || value === null || Number.isNaN(Number(value)) ? null : Number(value);
  return (
    <div className="rounded-lg border border-slate-100 bg-white p-2.5">
      <div className="text-xs">
        <div className="flex justify-between gap-3">
          <span className="font-medium text-slate-600">{label}</span>
          <span className="text-right font-semibold text-slate-900">{rate === null ? "Not available" : `${fmtNum(rate, 2)} / 1k`}</span>
        </div>
        <div className="mt-1 text-left text-[11px] text-slate-500">
          {pct === null ? "Percentile not available" : `Higher than ${Math.round(pct)}% of London wards`}
        </div>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct ?? 0}%`,
            background: pct === null ? "#e2e8f0" : "linear-gradient(90deg,#10b981,#f59e0b,#b91c1c)",
          }}
        />
      </div>
    </div>
  );
}


export function HoverRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-slate-400">{label}</span>
      <span className="text-right font-medium text-slate-700">{children}</span>
    </div>
  );
}
