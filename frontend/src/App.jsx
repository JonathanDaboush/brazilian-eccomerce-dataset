import { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import {
  buildEventBank,
  getHealth,
  getModels,
  getReplayStatus,
  getSummary,
  getTrainingRuns,
  getTrends,
  pauseReplay,
  predictModel,
  resetReplay,
  retrainModel,
  startReplay,
  stopReplay,
} from "./api";
import Toast, { useToasts } from "./Toast";
import UploadPage from "./UploadPage";
import AnalyticsPage from "./AnalyticsPage";
import "./App.css";

const POLL_MS = 5000;

const NAV_ITEMS = [
  { key: "dashboard", label: "📊 Dashboard" },
  { key: "analytics", label: "🔍 Analytics" },
  { key: "ml", label: "🤖 ML Predictions" },
  { key: "upload", label: "📤 Upload Data" },
  { key: "pipeline", label: "⚙️ Pipeline" },
];

function statusColor(val) {
  const v = String(val || "").toLowerCase();
  if (["healthy", "ok", "configured", "completed"].some(x => v.includes(x))) return "badge-green";
  if (["warning", "paused", "processing"].some(x => v.includes(x))) return "badge-yellow";
  if (["failed", "error", "unhealthy"].some(x => v.includes(x))) return "badge-red";
  return "badge-gray";
}

function StatusBadge({ label, value }) {
  return (
    <div className={`status-badge ${statusColor(value)}`}>
      <span className="badge-label">{label}</span>
      <span className="badge-value">{value || "Unknown"}</span>
    </div>
  );
}

function formatStatusLabel(status) {
  if (status === "failed") return "Failed";
  if (status === "running") return "Processing";
  if (status === "paused") return "Paused";
  if (status === "completed") return "Completed";
  return "Healthy";
}

function KpiCard({ label, value }) {
  return (
    <div className="kpi">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

// ─── ML Prediction Panel ───────────────────────────────────────────────────
function MLPanel({ models, trainingRuns, toast, refreshData }) {
  const [predictionModel, setPredictionModel] = useState("");
  const [featureValues, setFeatureValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);

  const availableModels = models.filter(m => m.available);
  const selectedModel = models.find(m => m.name === predictionModel);
  const schema = selectedModel?.feature_schema || [];

  useEffect(() => {
    if (availableModels.length && !predictionModel) {
      setPredictionModel(availableModels[0].name);
    }
  }, [availableModels]);

  const handleModelChange = name => {
    setPredictionModel(name);
    setFeatureValues({});
    setPrediction(null);
  };

  const handleFieldChange = (name, value) => {
    setFeatureValues(prev => ({ ...prev, [name]: value }));
  };

  const runPrediction = async () => {
    setLoading(true);
    try {
      const features = {};
      schema.forEach(f => {
        const raw = featureValues[f.name];
        if (raw !== undefined && raw !== "") {
          features[f.name] = f.numeric ? Number(raw) : raw;
        }
      });
      const res = await predictModel(predictionModel, features);
      setPrediction(res.data);
      toast.success("Prediction complete");
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const runRetrain = async () => {
    setRetraining(true);
    try {
      await retrainModel(predictionModel);
      await refreshData();
      toast.success("Model retraining complete");
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Retraining failed");
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div>
      <h2>ML Predictions</h2>
      <p className="subtitle">
        Select a model, optionally fill feature values (blanks use training-data defaults), then run.
      </p>
      <div className="ml-controls">
        <label>
          Model
          <select value={predictionModel} onChange={e => handleModelChange(e.target.value)}>
            {availableModels.map(m => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </select>
        </label>
        <button className="btn-primary" disabled={loading || !predictionModel} onClick={runPrediction}>
          {loading ? "Running…" : "Run Prediction"}
        </button>
        <button className="btn-secondary" disabled={retraining || !predictionModel || predictionModel === "product_recommendation"} onClick={runRetrain}>
          {retraining ? "Retraining…" : "Retrain Model"}
        </button>
      </div>

      {schema.length > 0 && (
        <div className="feature-form">
          <h4>Feature Inputs <span className="hint">(leave blank to use training-data default)</span></h4>
          <div className="feature-grid">
            {schema.map(f => (
              <label key={f.name} className="feature-field">
                <span className="field-name">{f.name}</span>
                <span className="field-type">{f.dtype}</span>
                {f.numeric ? (
                  <input
                    type="number"
                    placeholder={String(f.default ?? "")}
                    value={featureValues[f.name] ?? ""}
                    onChange={e => handleFieldChange(f.name, e.target.value)}
                  />
                ) : (
                  <input
                    type="text"
                    placeholder={String(f.default ?? "")}
                    value={featureValues[f.name] ?? ""}
                    onChange={e => handleFieldChange(f.name, e.target.value)}
                    list={`opts-${f.name}`}
                  />
                )}
                {f.sample_values?.length > 0 && (
                  <datalist id={`opts-${f.name}`}>
                    {f.sample_values.map(v => <option key={v} value={v} />)}
                  </datalist>
                )}
              </label>
            ))}
          </div>
        </div>
      )}

      {prediction && (
        <div className="prediction-result card">
          <h4>Result</h4>
          <pre>{JSON.stringify(prediction, null, 2)}</pre>
        </div>
      )}

      {models.length > 0 && (
        <div className="model-list">
          <h4>All Models</h4>
          <div className="model-grid">
            {models.map(m => (
              <div key={m.name} className={`model-card ${m.available ? "available" : "unavailable"}`}>
                <strong>{m.name}</strong>
                <span className={`badge ${m.available ? "badge-green" : "badge-gray"}`}>
                  {m.available ? "Available" : "No artifact"}
                </span>
                <span className={`badge ${m.training_data_available ? "badge-green" : "badge-gray"}`}>
                  {m.training_data_available ? "Training data ✓" : "No training data"}
                </span>
                {m.target_column && <span className="target-col">Target: {m.target_column}</span>}
                {m.trained_at && <span className="target-col">Last trained: {new Date(m.trained_at).toLocaleString()}</span>}
                {m.metrics && <pre className="metrics-preview">{JSON.stringify(m.metrics, null, 2)}</pre>}
              </div>
            ))}
          </div>
        </div>
      )}

      {trainingRuns.length > 0 && (
        <div className="card">
          <h4>Recent Training Runs</h4>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Trained</th>
                  <th>Metrics</th>
                </tr>
              </thead>
              <tbody>
                {trainingRuns.map(run => (
                  <tr key={run.artifact_path}>
                    <td>{run.model}</td>
                    <td>{run.trained_at ? new Date(run.trained_at).toLocaleString() : "—"}</td>
                    <td><pre className="metrics-preview">{JSON.stringify(run.metrics || {}, null, 2)}</pre></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Dashboard Section ─────────────────────────────────────────────────────
function DashboardSection({ health, replay, summary, trends, batchSize, setBatchSize, replaySpeed, setReplaySpeed, runAction, actionLoading, dateFrom, setDateFrom, dateTo, setDateTo, toast }) {
  const healthChips = useMemo(() => {
    if (!health) return [];
    return [
      ["API", health.api],
      ["Database", health.database],
      ["Kafka", health.kafka],
      ["Airflow", health.airflow],
      ["Producer", health.producer],
      ["Consumer", health.consumer],
      ["Replay", formatStatusLabel(health.replay?.status)],
    ];
  }, [health]);

  return (
    <>
      <section className="card">
        <h2>System Health</h2>
        <div className="badge-grid">
          {healthChips.map(([label, value]) => (
            <StatusBadge key={label} label={label} value={value} />
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Business KPIs</h2>
        <div className="kpi-grid">
          <KpiCard label="Revenue" value={`R$ ${summary?.kpis?.revenue?.toFixed(2) || "0.00"}`} />
          <KpiCard label="Active Orders" value={summary?.kpis?.active_orders ?? 0} />
          <KpiCard label="Delivered" value={summary?.kpis?.delivered_orders ?? 0} />
          <KpiCard label="Cancelled" value={summary?.kpis?.cancelled_orders ?? 0} />
          <KpiCard label="Avg Delivery (days)" value={summary?.kpis?.avg_delivery_days?.toFixed(2) || "0.00"} />
          <KpiCard label="Avg Review Score" value={summary?.kpis?.avg_review_score?.toFixed(2) || "0.00"} />
        </div>
      </section>

      <section className="card">
        <h2>Trend Charts</h2>
        <div className="date-range-row">
          <label>From <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} /></label>
          <label>To <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} /></label>
        </div>
        {trends.length === 0 ? (
          <p>No replayed trend data yet.</p>
        ) : (
          <>
            <h4>Revenue Over Time</h4>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trends} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `R$ ${Number(v).toFixed(2)}`} />
                <Line type="monotone" dataKey="revenue" stroke="#1b5cb4" dot={false} strokeWidth={2} name="Revenue" />
              </LineChart>
            </ResponsiveContainer>

            <h4 style={{ marginTop: "24px" }}>Delivered vs Cancelled</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={trends} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="delivered" fill="#22c55e" name="Delivered" />
                <Bar dataKey="cancelled" fill="#ef4444" name="Cancelled" />
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </section>

      <section className="card">
        <h2>Recent Activity</h2>
        <p><strong>Data Freshness:</strong> {summary?.data_freshness || "No processed events yet"}</p>
        <p><strong>Processing Rate:</strong> {summary?.processing_status?.processing_rate_eps ?? 0} events/sec</p>
        {summary?.processing_status?.last_error && <p><strong>Last Error:</strong> {summary.processing_status.last_error}</p>}
        <ul>
          {(summary?.recent_activity || []).map(row => (
            <li key={row.event_type}>{row.event_type}: {row.count_24h} events in last 24h</li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h2>Recent Processing Logs</h2>
        {summary?.recent_logs?.length ? (
          <div className="log-list">
            {summary.recent_logs.map((row, index) => (
              <div key={`${row.created_at}-${index}`} className="log-item">
                <div>
                  <strong>{row.event_type || "system"}</strong> · {row.status}
                </div>
                <div>{row.details}</div>
                <small>{row.created_at ? new Date(row.created_at).toLocaleString() : "—"}</small>
              </div>
            ))}
          </div>
        ) : (
          <p>No consumer logs yet.</p>
        )}
      </section>
    </>
  );
}

// ─── Pipeline Section ──────────────────────────────────────────────────────
function PipelineSection({ replay, batchSize, setBatchSize, replaySpeed, setReplaySpeed, runAction, actionLoading }) {
  return (
    <>
      <section className="card">
        <h2>Replay Controls</h2>
        <div className="controls">
          <label>
            Batch Size
            <input type="number" value={batchSize} min={1} max={5000} onChange={e => setBatchSize(Number(e.target.value))} />
          </label>
          <label>
            Speed (ms/event)
            <input type="number" value={replaySpeed} min={0} max={10000} onChange={e => setReplaySpeed(Number(e.target.value))} />
          </label>
          <button className="btn-primary" disabled={actionLoading} onClick={() => runAction(() => buildEventBank(), "Event bank prepared")}>
            Prepare Event Bank
          </button>
          <button className="btn-primary" disabled={actionLoading} onClick={() => runAction(() => startReplay({ batch_size: batchSize, replay_speed_ms: replaySpeed }), "Replay batch started")}>
            Start Batch
          </button>
          <button className="btn-secondary" disabled={actionLoading} onClick={() => runAction(() => pauseReplay(), "Paused")}>
            Pause
          </button>
          <button className="btn-secondary" disabled={actionLoading} onClick={() => runAction(() => stopReplay(), "Stopped")}>
            Stop
          </button>
          <button className="btn-danger" disabled={actionLoading} onClick={() => runAction(() => resetReplay(), "Replay reset")}>
            Reset
          </button>
        </div>
        <div className="status-box">
          <p><strong>Status:</strong> <span className={`badge ${statusColor(formatStatusLabel(replay?.status))}`}>{formatStatusLabel(replay?.status)}</span></p>
          <p><strong>Events Processed:</strong> {replay?.events_processed ?? 0}</p>
          <p><strong>Events Remaining:</strong> {replay?.events_remaining ?? 0}</p>
          <p><strong>Processing Rate:</strong> {replay?.processing_rate_eps ?? 0} events/sec</p>
          <p><strong>Latest Batch:</strong> {replay?.last_batch_produced ?? 0}</p>
          <p><strong>Failures:</strong> {replay?.events_failed ?? 0}</p>
          <p><strong>Producer Status:</strong> {replay?.latest_batch?.producer_status || "—"}</p>
          <p><strong>Last Error:</strong> {replay?.last_error || "None"}</p>
        </div>
      </section>

      <section className="card">
        <h2>Airflow DAGs</h2>
        <p>Two DAGs are configured and scheduled daily:</p>
        <div className="dag-list">
          <div className="dag-card">
            <strong>olist_replay_orchestration</strong>
            <p>Health check → Build event bank → Publish Kafka batch → Verify consumer → Update metrics → Summary</p>
            <span className="badge badge-green">@daily</span>
          </div>
          <div className="dag-card">
            <strong>ingest_ml_training_data</strong>
            <p>Health check → Check replay completion → Continue replay (if active) → Check new ML data → Trigger retrain → Verify artifacts → Summary</p>
            <span className="badge badge-green">@daily</span>
          </div>
        </div>
        <p style={{ marginTop: 12 }}>
          <a href="http://localhost:8080" target="_blank" rel="noreferrer" className="airflow-link">
            Open Airflow Console ↗
          </a>
        </p>
      </section>
    </>
  );
}

// ─── App ───────────────────────────────────────────────────────────────────
function App() {
  const [page, setPage] = useState("dashboard");
  const [health, setHealth] = useState(null);
  const [replay, setReplay] = useState(null);
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState([]);
  const [models, setModels] = useState([]);
  const [trainingRuns, setTrainingRuns] = useState([]);
  const [batchSize, setBatchSize] = useState(200);
  const [replaySpeed, setReplaySpeed] = useState(0);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const toast = useToasts();

  const loadDashboard = async () => {
    try {
      const [healthRes, replayRes, summaryRes, trendsRes, modelsRes, trainingRunsRes] = await Promise.all([
        getHealth(),
        getReplayStatus(),
        getSummary(),
        getTrends(dateFrom || undefined, dateTo || undefined),
        getModels(),
        getTrainingRuns(),
      ]);
      setHealth(healthRes.data);
      setReplay(replayRes.data);
      setSummary(summaryRes.data);
      setTrends(trendsRes.data?.series || []);
      setModels(modelsRes.data?.models || []);
      setTrainingRuns(trainingRunsRes.data?.items || []);
    } catch (err) {
      toast.error(err?.response?.data?.detail || err.message || "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();
    const timer = setInterval(loadDashboard, POLL_MS);
    return () => clearInterval(timer);
  }, [dateFrom, dateTo]);

  useEffect(() => {
    document.body.classList.toggle("dark", darkMode);
  }, [darkMode]);

  const runAction = async (action, successMsg) => {
    setActionLoading(true);
    try {
      await action();
      await loadDashboard();
      if (successMsg) toast.success(successMsg);
    } catch (err) {
      toast.error(err?.response?.data?.detail || err.message || "Action failed");
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner" />
        <p>Loading dashboard…</p>
      </div>
    );
  }

  return (
    <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <Toast toasts={toast.toasts} remove={toast.remove} />

      <nav className="sidebar">
        <div className="sidebar-header">
          <span className="brand">🛒 Olist</span>
          <button className="collapse-btn" onClick={() => setSidebarOpen(o => !o)}>
            {sidebarOpen ? "◀" : "▶"}
          </button>
        </div>
        {NAV_ITEMS.map(item => (
          <button
            key={item.key}
            className={`nav-item ${page === item.key ? "active" : ""}`}
            onClick={() => setPage(item.key)}
          >
            {item.label}
          </button>
        ))}
        <div className="sidebar-footer">
          <button className="theme-toggle" onClick={() => setDarkMode(d => !d)}>
            {darkMode ? "☀ Light" : "🌙 Dark"}
          </button>
        </div>
      </nav>

      <main className="main-content">
        <header className="top-bar">
          <h1>Brazilian E-commerce Operations</h1>
          <div className="top-bar-right">
            <span className="poll-label">Auto-refresh every 5s</span>
          </div>
        </header>

        <div className="page-body">
          {page === "dashboard" && (
            <DashboardSection
              health={health}
              replay={replay}
              summary={summary}
              trends={trends}
              batchSize={batchSize}
              setBatchSize={setBatchSize}
              replaySpeed={replaySpeed}
              setReplaySpeed={setReplaySpeed}
              runAction={runAction}
              actionLoading={actionLoading}
              dateFrom={dateFrom}
              setDateFrom={setDateFrom}
              dateTo={dateTo}
              setDateTo={setDateTo}
              toast={toast}
            />
          )}
          {page === "analytics" && <AnalyticsPage toast={toast} />}
          {page === "ml" && <MLPanel models={models} trainingRuns={trainingRuns} toast={toast} refreshData={loadDashboard} />}
          {page === "upload" && <UploadPage toast={toast} />}
          {page === "pipeline" && (
            <PipelineSection
              replay={replay}
              batchSize={batchSize}
              setBatchSize={setBatchSize}
              replaySpeed={replaySpeed}
              setReplaySpeed={setReplaySpeed}
              runAction={runAction}
              actionLoading={actionLoading}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
