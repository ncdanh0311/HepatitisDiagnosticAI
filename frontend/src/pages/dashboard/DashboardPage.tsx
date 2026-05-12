// frontend/src/pages/dashboard/DashboardPage.tsx
import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, PieChart, Pie, Cell, Tooltip,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid,
} from "recharts";
import { Activity, Users, AlertTriangle, TrendingUp } from "lucide-react";
import { analyticsApi } from "@/api/index";

const COLORS = ["#00D4FF", "#0072FF", "#F59E0B", "#EF4444", "#8B5CF6"];

export default function DashboardPage() {
  const { data: kpis } = useQuery({ queryKey: ["dashboard"], queryFn: () => analyticsApi.dashboard().then((r) => r.data) });
  const { data: dist } = useQuery({ queryKey: ["distribution"], queryFn: () => analyticsApi.distribution().then((r) => r.data) });
  const { data: trends } = useQuery({ queryKey: ["trends"], queryFn: () => analyticsApi.trends(30).then((r) => r.data) });

  const kpiCards = [
    { icon: Users,         label: "Total Patients",     value: kpis?.total_patients ?? "—",      color: "text-brand-400" },
    { icon: Activity,      label: "Total Predictions",  value: kpis?.total_predictions ?? "—",   color: "text-success" },
    { icon: TrendingUp,    label: "Last 30 Days",       value: kpis?.recent_predictions_30d ?? "—", color: "text-warning" },
    { icon: AlertTriangle, label: "High Risk Cases",    value: kpis?.high_risk_count ?? "—",     color: "text-danger" },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-muted text-sm mt-1">Hepatitis AI Analytics Overview</p>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        {kpiCards.map(({ icon: Icon, label, value, color }) => (
          <div key={label} className="kpi-card">
            <div className={`${color} mb-1`}><Icon className="w-5 h-5" /></div>
            <div className="kpi-value">{typeof value === "number" ? value.toLocaleString() : value}</div>
            <div className="kpi-label">{label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Trend chart */}
        <div className="card">
          <h3 className="font-semibold text-white mb-4">Prediction Trends (30 days)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={trends ?? []}>
              <defs>
                <linearGradient id="brand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00D4FF" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00D4FF" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
              <XAxis dataKey="date" tick={{ fill: "#8892A4", fontSize: 11 }} />
              <YAxis tick={{ fill: "#8892A4", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#1C2333", border: "1px solid #ffffff20", borderRadius: 8 }} />
              <Area type="monotone" dataKey="count" stroke="#00D4FF" fill="url(#brand)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Distribution pie */}
        <div className="card">
          <h3 className="font-semibold text-white mb-4">Disease Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={dist ?? []} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                   dataKey="count" nameKey="category" paddingAngle={3}>
                {(dist ?? []).map((_: unknown, i: number) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "#1C2333", border: "1px solid #ffffff20", borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 mt-2">
            {(dist ?? []).map((d: { category: string }, i: number) => (
              <div key={d.category} className="flex items-center gap-1.5 text-xs text-muted">
                <div className="w-2 h-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                {d.category}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
