import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import {
  fetchDashboard,
  fetchFeatureSummary,
  fetchHealth,
  fetchModels,
  fetchPrediction,
  fetchReplayStatus,
  resetReplay,
  runReplay,
  uploadFile,
} from "./api";

const REPLAY_OPTIONS = [100, 500, 1000];
const PACE_OPTIONS = [
  { label: "Full speed", value: 0 },
  { label: "Slow (50 ms/event)", value: 50 },
  { label: "Very slow (200 ms/event)", value: 200 },
];

function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value || 0);
}

function fmt(value, decimals = 1) {
  if (value == null) return "n/a";
  return typeof value === "number" ? value.toFixed(decimals) : String(value);
}

function StatusBadge({ status }) {
  const map = {
    delivered: "badge-green",
    cancelled: "badge-red",
    shipped: "badge-blue",
    approved: "badge-blue",
    created: "badge-gray",
    invoiced: "badge-gray",
    unavailable: "badge-gray",
    processing: "badge-gray",
  };
  const cls = map[status] || "badge-gray";
  return <span className={`badge ${cls}`}>{status || "unknown"}</span>;
}

function App() {
  const [dashboard, setDashboard] = useState(null);
  const [health, setHealth] = useState(null);
  const [replay, setReplay] = useState(null);
  const [models, setModels] = useState([]);
  const [features, setFeatures] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionModel, setPredictionModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [pace, setPace] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef(null);

  const loadPage = useCallback(async () => {
    try {
      setError("");
      const [dashboardData, replayData, healthData, modelsData, featuresData] = await Promise.all([
        fetchDashboard(),
        fetchReplayStatus(),
        fetchHealth(),
        fetchModels(),
        fetchFeatureSummary(),
      ]);
      setDashboard(dashboardData);
      setReplay(replayData);
      setHealth(healthData);
      setModels(modelsData);
      setFeatures(featuresData);
      if (!predictionModel && modelsData.length > 0) {
        const firstPredictable = modelsData.find((model) => model.supports_prediction);
        if (firstPredictable) {
          setPredictionModel(firstPredictable.model_name);
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to load dashboard.");
    } finally {
      setLoading(false);
    }
  }, [predictionModel]);

  useEffect(() => {
    loadPage();
    const interval = setInterval(loadPage, 15000);
    return () => clearInterval(interval);
  }, [loadPage]);

  const handleReplay = async (limit) => {
    setActionLoading(true);
    setSuccess("");
    try {
      const result = await runReplay(limit, pace);
      setSuccess(
        `Replayed ${result.processed_events} new events (${result.duplicate_events} duplicates skipped).`
      );
      await loadPage();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Replay failed.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleReset = async () => {
    setActionLoading(true);
    setSuccess("");
    try {
      await resetReplay();
      setPrediction(null);
      setSuccess("Live state reset. All processed events cleared.");
      await loadPage();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Reset failed.");
    } finally {
      setActionLoading(false);
    }
  };

  const handlePrediction = async () => {
    if (!predictionModel) return;
    setActionLoading(true);
    try {
      const data = await fetchPrediction(predictionModel, 0);
      setPrediction(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Prediction failed.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadResult(null);
    setUploadError("");
    try {
      const result = await uploadFile(file);
      setUploadResult(result);
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || "Upload failed.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const kpis = useMemo(() => dashboard?.kpis || {}, [dashboard]);

  if (loading) {
    return <main className="page"><section className="empty-state">Loading dashboard…</section></main>;
  }

  return (
    <main className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">Brazilian E-commerce · Olist Dataset</p>
          <h1>Business Operations Dashboard</h1>
          <p className="hero-copy">
            Replay historical Olist order events into the live store, monitor KPI changes in real time,
            inspect ML model predictions, and upload data files for validation.
          </p>
        </div>
        <div className="hero-actions">
          <div className="pace-control">
            <label htmlFor="pace-select">Replay speed</label>
            <select
              id="pace-select"
              value={pace}
              onChange={(e) => setPace(Number(e.target.value))}
              disabled={actionLoading}
            >
              {PACE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          {REPLAY_OPTIONS.map((limit) => (
            <button key={limit} onClick={() => handleReplay(limit)} disabled={actionLoading}>
              Replay {limit} events
            </button>
          ))}
          <button className="secondary" onClick={handleReset} disabled={actionLoading}>
            Reset live state
          </button>
        </div>
      </section>

      {error ? <section className="banner error">{error}</section> : null}
      {success ? <section className="banner success">{success}</section> : null}

      <section className="grid kpi-grid">
        <KpiCard label="Orders processed" value={kpis.orders ?? 0} />
        <KpiCard label="Total revenue" value={currency(kpis.revenue)} />
        <KpiCard label="Delivered orders" value={kpis.delivered_orders ?? 0} />
        <KpiCard label="Cancelled orders" value={kpis.cancelled_orders ?? 0} />
        <KpiCard label="Avg order value" value={currency(kpis.avg_order_value)} />
        <KpiCard label="Unique customers" value={kpis.unique_customers ?? 0} />
        <KpiCard
          label="Avg delivery time"
          value={kpis.avg_delivery_days != null ? `${fmt(kpis.avg_delivery_days)} days` : "n/a"}
        />
        <KpiCard
          label="Customer satisfaction"
          value={kpis.avg_satisfaction_score != null ? `${fmt(kpis.avg_satisfaction_score, 2)} / 5` : "n/a"}
        />
      </section>

      <section className="grid two-column">
        <Panel title="System status" subtitle="Backend connectivity and event-bank totals">
          {health ? (
            <dl className="meta-list">
              <MetaItem label="Database" value={health.database_url === "sqlite" ? "SQLite (local)" : "MySQL (Docker)"} />
              <MetaItem label="Source data" value={health.source_data_present ? "✅ Available" : "❌ Missing"} />
              <MetaItem label="Immutable events" value={health.event_bank?.event_count ?? 0} />
              <MetaItem label="Event date range" value={`${health.event_bank?.date_range?.start?.slice(0, 10) || "n/a"} → ${health.event_bank?.date_range?.end?.slice(0, 10) || "n/a"}`} />
              <MetaItem label="Processed events" value={health.database_counts?.processed_events ?? 0} />
              <MetaItem label="Live orders" value={health.database_counts?.orders ?? 0} />
              <MetaItem label="Replay batches run" value={health.database_counts?.replay_batches ?? 0} />
            </dl>
          ) : (
            <Empty message="Health data unavailable." />
          )}
        </Panel>

        <Panel title="Replay status" subtitle="Recent batch history">
          {replay?.recent_batches?.length ? (
            <table>
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Status</th>
                  <th>Processed</th>
                  <th>Completed</th>
                </tr>
              </thead>
              <tbody>
                {replay.recent_batches.slice(0, 5).map((batch) => (
                  <tr key={batch.id}>
                    <td>#{batch.id}</td>
                    <td><StatusBadge status={batch.status === "completed" ? "delivered" : batch.status === "failed" ? "cancelled" : "processing"} /></td>
                    <td>{batch.processed_events ?? 0}</td>
                    <td>{batch.completed_at ? batch.completed_at.slice(0, 19).replace("T", " ") : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty message="No replay batches run yet. Use the controls above to start." />
          )}
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="Monthly trend" subtitle="Order volume and revenue by month">
          <TrendTable rows={dashboard?.trends || []} />
        </Panel>
        <Panel title="Top product categories" subtitle="Revenue by translated product category">
          <CategoryTable rows={dashboard?.top_categories || []} />
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="Recent processing activity" subtitle="Most recently processed order events">
          <ActivityTable rows={dashboard?.activity || []} />
        </Panel>
        <Panel title="Recent orders" subtitle="Live order records in the operational database">
          <OrdersTable rows={dashboard?.recent_orders || []} />
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="Top customers" subtitle="Customers by lifetime revenue (from processed events)">
          {features?.top_customers?.length ? (
            <table>
              <thead>
                <tr>
                  <th>Customer</th>
                  <th>State</th>
                  <th>Orders</th>
                  <th>Lifetime revenue</th>
                </tr>
              </thead>
              <tbody>
                {features.top_customers.map((row) => (
                  <tr key={row.customer_id}>
                    <td>{row.customer_id}</td>
                    <td>{row.state || "n/a"}</td>
                    <td>{row.order_count}</td>
                    <td>{currency(row.lifetime_revenue)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty message="Replay events to populate the customer feature store." />
          )}
        </Panel>

        <Panel title="Top sellers" subtitle="Sellers by revenue from processed order items">
          {features?.top_sellers?.length ? (
            <table>
              <thead>
                <tr>
                  <th>Seller</th>
                  <th>State</th>
                  <th>Orders</th>
                  <th>Revenue</th>
                </tr>
              </thead>
              <tbody>
                {features.top_sellers.map((row) => (
                  <tr key={row.seller_id}>
                    <td>{row.seller_id}</td>
                    <td>{row.state || "n/a"}</td>
                    <td>{row.order_count}</td>
                    <td>{currency(row.revenue)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty message="Replay events to populate the seller feature store." />
          )}
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="ML models" subtitle="Saved model artifacts and sample prediction access">
          <div className="ml-controls">
            <select value={predictionModel} onChange={(event) => setPredictionModel(event.target.value)}>
              <option value="">Select a model</option>
              {models
                .filter((model) => model.supports_prediction)
                .map((model) => (
                  <option key={model.model_name} value={model.model_name}>
                    {model.model_name.replace(/_/g, " ")}
                  </option>
                ))}
            </select>
            <button onClick={handlePrediction} disabled={!predictionModel || actionLoading}>
              Run sample prediction
            </button>
          </div>
          <ModelTable rows={models} />
        </Panel>

        <Panel title="Prediction result" subtitle="Inference on a preserved training sample">
          {prediction ? (
            <dl className="meta-list">
              <MetaItem label="Model" value={prediction.model_name.replace(/_/g, " ")} />
              <MetaItem label="Prediction" value={String(prediction.prediction)} />
              <MetaItem label="Observed target" value={String(prediction.target)} />
              <MetaItem label="Features used" value={prediction.feature_count} />
              <MetaItem
                label="Model score"
                value={prediction.score ? JSON.stringify(prediction.score) : "n/a"}
              />
              {prediction.probabilities ? (
                <MetaItem
                  label="Class probabilities"
                  value={prediction.probabilities.map((p) => (p * 100).toFixed(1) + "%").join(" / ")}
                />
              ) : null}
            </dl>
          ) : (
            <Empty message="Select a supported model above and click 'Run sample prediction'." />
          )}
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="Upload data file" subtitle="Validate a CSV or Excel file — originals are never modified">
          <div className="upload-area">
            <p className="upload-hint">
              Upload a <strong>.csv</strong>, <strong>.xls</strong>, or <strong>.xlsx</strong> file
              to validate its structure and preview the first rows. The original file is never
              stored or modified by the system.
            </p>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.xls,.xlsx"
              onChange={handleUpload}
              disabled={uploading}
              className="file-input"
            />
            {uploading ? <p className="upload-status">Uploading and validating…</p> : null}
            {uploadError ? <p className="upload-error">{uploadError}</p> : null}
          </div>
          {uploadResult ? (
            <dl className="meta-list" style={{ marginTop: "16px" }}>
              <MetaItem label="File" value={uploadResult.filename} />
              <MetaItem label="Rows" value={uploadResult.rows} />
              <MetaItem label="Columns" value={uploadResult.column_count} />
              <MetaItem label="Validation" value={uploadResult.validation} />
              <MetaItem label="Columns with nulls" value={uploadResult.null_columns?.join(", ") || "none"} />
              <MetaItem label="Note" value={uploadResult.note} />
            </dl>
          ) : null}
        </Panel>

        <Panel title="Upload preview" subtitle="First rows of the validated file">
          {uploadResult?.preview?.length ? (
            <div className="preview-scroll">
              <table>
                <thead>
                  <tr>
                    {uploadResult.columns.slice(0, 6).map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                    {uploadResult.columns.length > 6 ? <th>…</th> : null}
                  </tr>
                </thead>
                <tbody>
                  {uploadResult.preview.map((row, idx) => (
                    <tr key={idx}>
                      {uploadResult.columns.slice(0, 6).map((col) => (
                        <td key={col}>{row[col] == null ? "—" : String(row[col])}</td>
                      ))}
                      {uploadResult.columns.length > 6 ? <td>…</td> : null}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty message="Upload a CSV or Excel file on the left to see a preview here." />
          )}
        </Panel>
      </section>
    </main>
  );
}

function Panel({ title, subtitle, children }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function KpiCard({ label, value }) {
  return (
    <article className="kpi-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function MetaItem({ label, value }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}

function Empty({ message }) {
  return <div className="empty-state">{message}</div>;
}

function TrendTable({ rows }) {
  if (!rows.length) {
    return <Empty message="Run a replay batch to populate trend data." />;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Month</th>
          <th>Orders</th>
          <th>Revenue</th>
          <th>Delivered</th>
          <th>Cancelled</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.period}>
            <td>{row.period}</td>
            <td>{row.orders}</td>
            <td>{currency(row.revenue)}</td>
            <td>{row.delivered}</td>
            <td>{row.cancelled}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CategoryTable({ rows }) {
  if (!rows.length) {
    return <Empty message="No category revenue available yet." />;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Revenue</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.category}>
            <td>{row.category}</td>
            <td>{currency(row.revenue)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ActivityTable({ rows }) {
  if (!rows.length) {
    return <Empty message="No processing activity yet." />;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Event type</th>
          <th>Order ID</th>
          <th>Status</th>
          <th>Processed at</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.event_id}>
            <td>{row.event_type}</td>
            <td>{row.order_id?.slice(0, 8)}…</td>
            <td><StatusBadge status={row.status} /></td>
            <td>{row.processed_at ? row.processed_at.slice(0, 19).replace("T", " ") : "pending"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function OrdersTable({ rows }) {
  if (!rows.length) {
    return <Empty message="Replay orders to inspect live order state." />;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Order ID</th>
          <th>Status</th>
          <th>State</th>
          <th>Revenue</th>
          <th>Purchased</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.order_id}>
            <td>{row.order_id?.slice(0, 8)}…</td>
            <td><StatusBadge status={row.status} /></td>
            <td>{row.customer_state || "n/a"}</td>
            <td>{currency(row.payment_value || row.revenue)}</td>
            <td>{row.purchase_ts ? String(row.purchase_ts).slice(0, 10) : "n/a"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ModelTable({ rows }) {
  if (!rows.length) {
    return <Empty message="No saved model artifacts found." />;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Prediction</th>
          <th>Features</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.model_name}>
            <td>{row.model_name.replace(/_/g, " ")}</td>
            <td>{row.supports_prediction ? "✅ available" : "artifact only"}</td>
            <td>{row.feature_count ?? "n/a"}</td>
            <td>{row.score ? JSON.stringify(row.score) : row.error || "n/a"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default App;
