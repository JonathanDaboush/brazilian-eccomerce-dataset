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
export const getTrends = (dateFrom, dateTo) => {
  const params = {};
  if (dateFrom) params.date_from = dateFrom;
  if (dateTo) params.date_to = dateTo;
  return api.get("/dashboard/trends", { params });
};
export const getModels = () => api.get("/ml/models");
export const getTrainingRuns = (limit = 20) =>
  api.get("/ml/training-runs", { params: { limit } });
export const predictModel = (modelName, features = {}) =>
  api.post(`/ml/predict/${modelName}`, { features });
export const retrainModel = modelName =>
  api.post("/ml/retrain", modelName ? { model_name: modelName } : {});
export const getCustomerFeatures = (limit = 50) =>
  api.get("/features/customer", { params: { limit } });
export const getSellerFeatures = (limit = 50) =>
  api.get("/features/seller", { params: { limit } });
export const getProductFeatures = (limit = 50) =>
  api.get("/features/product", { params: { limit } });
export const validateTrainingCsv = formData =>
  api.post("/upload/validate-training-csv", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 30000,
  });
export const ingestTrainingCsv = formData =>
  api.post("/upload/training-csv", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 30000,
  });
export const getDatasetSchemas = () => api.get("/upload/dataset-schemas");
