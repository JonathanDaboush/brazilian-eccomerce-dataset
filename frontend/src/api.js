import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 30000,
});

export async function fetchDashboard() {
  const { data } = await api.get("/api/dashboard");
  return data;
}

export async function fetchReplayStatus() {
  const { data } = await api.get("/api/replay");
  return data;
}

export async function fetchHealth() {
  const { data } = await api.get("/api/health");
  return data;
}

export async function fetchModels() {
  const { data } = await api.get("/api/ml/models");
  return data.models;
}

export async function fetchFeatureSummary() {
  const { data } = await api.get("/api/features");
  return data;
}

export async function runReplay(limit, paceMs = 0) {
  const { data } = await api.post("/api/replay", {
    start_offset: 0,
    limit,
    pace_ms: paceMs,
  });
  return data;
}

export async function resetReplay() {
  const { data } = await api.post("/api/replay/reset");
  return data;
}

export async function fetchPrediction(modelName, rowIndex = 0) {
  const { data } = await api.get(`/api/ml/predict/${modelName}`, {
    params: { row_index: rowIndex },
  });
  return data;
}

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/api/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60000,
  });
  return data;
}
