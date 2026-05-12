// frontend/src/components/layout/Sidebar.tsx
import { NavLink } from "react-router-dom";
import {
  Activity, BarChart3, Bot, FlaskConical,
  LayoutDashboard, LogOut, Users, Stethoscope,
} from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/dashboard",    icon: LayoutDashboard, label: "Dashboard" },
  { to: "/patients",     icon: Users,           label: "Patients" },
  { to: "/predictions",  icon: Stethoscope,     label: "Predictions" },
  { to: "/analytics",   icon: BarChart3,        label: "Analytics" },
  { to: "/chatbot",      icon: Bot,             label: "AI Chatbot" },
  { to: "/admin",        icon: FlaskConical,    label: "MLOps", roles: ["admin"] },
];

export function Sidebar() {
  const { user, clearAuth } = useAuthStore();

  return (
    <aside className="fixed inset-y-0 left-0 w-64 bg-surface-1 border-r border-white/10 flex flex-col z-40">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Activity className="text-brand-400 w-7 h-7" />
          <div>
            <p className="font-bold text-white text-sm">Hepatitis AI</p>
            <p className="text-muted text-xs">Healthcare Platform</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV_ITEMS.filter((item) => {
          if (!item.roles) return true;
          return item.roles.includes(user?.role ?? "");
        }).map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx("nav-link", isActive && "active")
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* User card */}
      <div className="px-3 pb-4">
        <div className="card p-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-brand-gradient flex items-center justify-center text-white text-xs font-bold">
            {user?.full_name?.[0] ?? "?"}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-white truncate">{user?.full_name}</p>
            <p className="text-xs text-muted capitalize">{user?.role}</p>
          </div>
          <button onClick={clearAuth} title="Logout" className="text-muted hover:text-danger transition-colors">
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}
