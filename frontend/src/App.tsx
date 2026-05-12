// frontend/src/App.tsx
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Sidebar } from "@/components/layout/Sidebar";
import { useAuthStore } from "@/store/authStore";
import DashboardPage from "@/pages/dashboard/DashboardPage";
import LoginPage from "@/pages/auth/LoginPage";
const PatientsPage  = () => <div className="text-white p-8"><h1 className="text-2xl font-bold">Patients</h1></div>;
const PredictPage   = () => <div className="text-white p-8"><h1 className="text-2xl font-bold">Predictions</h1></div>;
const AnalyticsPage = () => <div className="text-white p-8"><h1 className="text-2xl font-bold">Analytics</h1></div>;
const ChatbotPage   = () => <div className="text-white p-8"><h1 className="text-2xl font-bold">AI Chatbot</h1></div>;
const AdminPage     = () => <div className="text-white p-8"><h1 className="text-2xl font-bold">MLOps Admin</h1></div>;

const qc = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 60_000 } },
});

function AuthGuard({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((s) => s.accessToken);
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 ml-64 overflow-y-auto p-8 bg-surface">
        {children}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/*"
            element={
              <AuthGuard>
                <AppLayout>
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard"   element={<DashboardPage />} />
                    <Route path="/patients"    element={<PatientsPage />} />
                    <Route path="/predictions" element={<PredictPage />} />
                    <Route path="/analytics"   element={<AnalyticsPage />} />
                    <Route path="/chatbot"     element={<ChatbotPage />} />
                    <Route path="/admin"       element={<AdminPage />} />
                  </Routes>
                </AppLayout>
              </AuthGuard>
            }
          />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
