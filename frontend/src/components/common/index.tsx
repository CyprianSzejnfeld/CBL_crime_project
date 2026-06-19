import type { ReactNode } from "react";

export function MetricCard({
  label,
  value,
  sub,
  accent,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  accent?: string;
}) {
  return (
    <div className="card p-4">
      <div className="stat-label">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-900" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex h-full min-h-[180px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white/60 p-6 text-center">
      <div className="text-sm font-medium text-slate-600">{title}</div>
      {hint && <div className="mt-1 text-xs text-slate-400">{hint}</div>}
    </div>
  );
}

export function LoadingSkeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-slate-200/70 ${className}`} />;
}
