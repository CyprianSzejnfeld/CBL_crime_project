import { NavLink } from "react-router-dom";
import { LayoutDashboard, Map, Search } from "lucide-react";

const NAV = [
  { to: "/", label: "Package allocation", icon: LayoutDashboard, end: true },
  { to: "/reductions", label: "Stop & search review", icon: Search },
  { to: "/clusters", label: "Fairness diagnostics", icon: Map },
];

export function Sidebar() {
  return (
    <nav className="flex w-full shrink-0 gap-1 overflow-x-auto border-b border-slate-200 bg-white px-3 py-2 md:w-56 md:flex-col md:border-b-0 md:border-r md:py-4">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) =>
            `flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
              isActive ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
            }`
          }
        >
          <Icon className="h-4 w-4" />
          {label}
        </NavLink>
      ))}
    </nav>
  );
}
