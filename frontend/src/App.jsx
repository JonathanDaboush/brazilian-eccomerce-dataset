import { useEffect, useMemo, useState } from "react";
import {
  buildEventBank,
  getHealth,
  getModels,
  getReplayStatus,
  getSummary,
  getTrends,
  pauseReplay,
  predictModel,
  resetReplay,
  startReplay,
  stopReplay,
} from "./api";
import "./App.css";

const POLL_MS = 5000;

function formatStatusLabel(status) {
  if (status === "failed") return "Failed";
  if (status === "running") return "Processing";
  if (status === "paused") return "Paused";
  if (status === "completed") return "Completed";
  return "Healthy";
}

function App() {
  const [health, setHealth] = useState(null);
  const [replay, setReplay] = useState(null);
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState([]);
  const [models, setModels] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [predictionModel, setPredictionModel] = useState("delivery_delay");
  const [batchSize, setBatchSize] = useState(200);
  const [replaySpeed, setReplaySpeed] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);

  const loadDashboard = async () => {
    try {
      const [healthRes, replayRes, summaryRes, trendsRes, modelsRes] = await Promise.all([
        getHealth(),
        getReplayStatus(),
        getSummary(),
        getTrends(),
        getModels(),
      ]);
      setHealth(healthRes.data);
      setReplay(replayRes.data);
      setSummary(summaryRes.data);
      setTrends(trendsRes.data?.series || []);
      setModels(modelsRes.data?.models || []);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Failed to load dashboard data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();
    const timer = setInterval(loadDashboard, POLL_MS);
    return () => clearInterval(timer);
  }, []);

  const runAction = async action => {
    setActionLoading(true);
    try {
      await action();
      await loadDashboard();
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Action failed");
    } finally {
      setActionLoading(false);
    }
  };

  const runPrediction = async () => {
    setActionLoading(true);
    try {
      const res = await predictModel(predictionModel, {});
      setPrediction(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Prediction failed");
    } finally {
      setActionLoading(false);
    }
  };

  const healthChips = useMemo(() => {
    if (!health) return [];
    return [
      ["API", health.api],
      ["Database", health.database],
      ["Kafka", health.kafka],
      ["Airflow", health.airflow],
      ["Replay", formatStatusLabel(health.replay?.status)],
    ];
  }, [health]);

  if (loading) return <main className="page"><h2>Loading dashboard...</h2></main>;

  return (
    <main className="page">
      <header>
        <h1>Brazilian E-commerce Operations Dashboard</h1>
        <p>Real replay pipeline status, business metrics, and ML outputs.</p>
      </header>

      {error && <section className="card error">{error}</section>}

      <section className="card">
        <h2>System Health</h2>
        <div className="chip-grid">
          {healthChips.map(([label, value]) => (
            <div className="chip" key={label}><strong>{label}:</strong> {value || "Unknown"}</div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Replay Controls</h2>
        <div className="controls">
          <label>
            Batch Size
            <input type="number" value={batchSize} min={1} max={5000} onChange={e => setBatchSize(Number(e.target.value))} />
          </label>
          <label>
            Replay Speed (ms/event)
            <input type="number" value={replaySpeed} min={0} max={10000} onChange={e => setReplaySpeed(Number(e.target.value))} />
          </label>
          <button disabled={actionLoading} onClick={() => runAction(() => buildEventBank())}>Prepare Immutable Event Bank</button>
          <button disabled={actionLoading} onClick={() => runAction(() => startReplay({ batch_size: batchSize, replay_speed_ms: replaySpeed }))}>Start Replay Batch</button>
          <button disabled={actionLoading} onClick={() => runAction(() => pauseReplay())}>Pause</button>
          <button disabled={actionLoading} onClick={() => runAction(() => stopReplay())}>Stop</button>
          <button disabled={actionLoading} onClick={() => runAction(() => resetReplay())}>Reset Replay State</button>
        </div>
        <div className="status-box">
          <p><strong>Status:</strong> {formatStatusLabel(replay?.status)}</p>
          <p><strong>Events Processed:</strong> {replay?.events_processed ?? 0}</p>
          <p><strong>Events Remaining:</strong> {replay?.events_remaining ?? 0}</p>
          <p><strong>Latest Batch Size:</strong> {replay?.last_batch_produced ?? 0}</p>
          <p><strong>Failures:</strong> {replay?.events_failed ?? 0}</p>
        </div>
      </section>

      <section className="card">
        <h2>Business KPIs</h2>
        <div className="kpi-grid">
          <div className="kpi"><span>Revenue</span><strong>R$ {summary?.kpis?.revenue?.toFixed(2) || "0.00"}</strong></div>
          <div className="kpi"><span>Active Orders</span><strong>{summary?.kpis?.active_orders ?? 0}</strong></div>
          <div className="kpi"><span>Delivered</span><strong>{summary?.kpis?.delivered_orders ?? 0}</strong></div>
          <div className="kpi"><span>Cancelled</span><strong>{summary?.kpis?.cancelled_orders ?? 0}</strong></div>
          <div className="kpi"><span>Avg Delivery (days)</span><strong>{summary?.kpis?.avg_delivery_days?.toFixed(2) || "0.00"}</strong></div>
          <div className="kpi"><span>Avg Review Score</span><strong>{summary?.kpis?.avg_review_score?.toFixed(2) || "0.00"}</strong></div>
        </div>
      </section>

      <section className="card">
        <h2>30-Day Trend Snapshot</h2>
        {trends.length === 0 ? <p>No replayed trend data yet.</p> : (
          <table>
            <thead>
              <tr><th>Date</th><th>Revenue</th><th>Delivered</th><th>Cancelled</th></tr>
            </thead>
            <tbody>
              {trends.slice(-10).map(day => (
                <tr key={day.day} title={`Revenue ${day.revenue}`}>
                  <td>{day.day}</td>
                  <td>R$ {day.revenue.toFixed(2)}</td>
                  <td>{day.delivered}</td>
                  <td>{day.cancelled}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h2>ML Predictions</h2>
        <div className="controls">
          <label>
            Model
            <select value={predictionModel} onChange={e => setPredictionModel(e.target.value)}>
              {models.filter(m => m.available).map(m => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </label>
          <button disabled={actionLoading} onClick={runPrediction}>Run Sample Prediction</button>
        </div>
        {prediction && <pre>{JSON.stringify(prediction, null, 2)}</pre>}
      </section>

      <section className="card">
        <h2>Recent Activity & Data Freshness</h2>
        <p><strong>Data Freshness:</strong> {summary?.data_freshness || "No processed events yet"}</p>
        <ul>
          {(summary?.recent_activity || []).map(row => (
            <li key={row.event_type}>{row.event_type}: {row.count_24h} events in last 24h</li>
          ))}
        </ul>
      </section>
    </main>
  );
}

export default App;
