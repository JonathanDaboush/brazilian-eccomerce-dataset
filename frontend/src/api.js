import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 15000,
});

export const getHealth = () => api.get("/health/system");
export const buildEventBank = () => api.post("/replay/event-bank/build");
export const getReplayStatus = () => api.get("/replay/status");
export const startReplay = payload => api.post("/replay/start", payload);
export const pauseReplay = () => api.post("/replay/pause");
export const stopReplay = () => api.post("/replay/stop");
export const resetReplay = () => api.post("/replay/reset");
export const getSummary = () => api.get("/dashboard/summary");
export const getTrends = () => api.get("/dashboard/trends");
export const getModels = () => api.get("/ml/models");
export const predictModel = (modelName, features = {}) => api.post(`/ml/predict/${modelName}`, { features });
