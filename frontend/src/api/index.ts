// frontend/src/api/auth.ts
import { apiClient } from "./client";

export const authApi = {
  login: (email: string, password: string) =>
    apiClient.post("/auth/login", { email, password }),

  register: (email: string, full_name: string, password: string, role = "researcher") =>
    apiClient.post("/auth/register", { email, full_name, password, role }),

  me: () => apiClient.get("/auth/me"),

  logout: () => apiClient.post("/auth/logout"),
};

// frontend/src/api/patients.ts
export const patientsApi = {
  list: (page = 1, size = 20, search = "") =>
    apiClient.get("/patients", { params: { page, size, search } }),

  create: (data: object) => apiClient.post("/patients", data),
  get: (id: string) => apiClient.get(`/patients/${id}`),
  update: (id: string, data: object) => apiClient.put(`/patients/${id}`, data),
  delete: (id: string) => apiClient.delete(`/patients/${id}`),
  getRecords: (id: string) => apiClient.get(`/patients/${id}/records`),
  addRecord: (id: string, data: object) => apiClient.post(`/patients/${id}/records`, data),
  getPredictions: (id: string) => apiClient.get("/predictions", { params: { patient_id: id } }),
};

// frontend/src/api/predictions.ts
export const predictionsApi = {
  create: (data: object) => apiClient.post("/predictions", data),
  list: (patientId?: string) =>
    apiClient.get("/predictions", { params: patientId ? { patient_id: patientId } : {} }),
};

// frontend/src/api/analytics.ts
export const analyticsApi = {
  dashboard: () => apiClient.get("/analytics/dashboard"),
  distribution: () => apiClient.get("/analytics/disease-distribution"),
  trends: (days = 30) => apiClient.get("/analytics/prediction-trends", { params: { days } }),
};

// frontend/src/api/chatbot.ts
export const chatbotApi = {
  send: (session_id: string, message: string) =>
    apiClient.post("/chat/message", { session_id, message }),
  getSession: (session_id: string) => apiClient.get(`/chat/sessions/${session_id}`),
};
