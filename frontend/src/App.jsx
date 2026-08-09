import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import {
  fetchDashboard,
  fetchHealth,
  fetchModels,
  fetchPrediction,
  fetchReplayStatus,
  resetReplay,
  runReplay,
} from "./api";

const REPLAY_OPTIONS = [100, 500, 1000];

function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value || 0);
}

function App() {
  const [dashboard, setDashboard] = useState(null);
  const [health, setHealth] = useState(null);
  const [replay, setReplay] = useState(null);
  const [models, setModels] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [predictionModel, setPredictionModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState("");

  const loadPage = useCallback(async () => {
    try {
      setError("");
      const [dashboardData, replayData, healthData, modelsData] = await Promise.all([
        fetchDashboard(),
        fetchReplayStatus(),
        fetchHealth(),
        fetchModels(),
      ]);
      setDashboard(dashboardData);
      setReplay(replayData);
      setHealth(healthData);
      setModels(modelsData);
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
    try {
      await runReplay(limit);
      await loadPage();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Replay failed.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleReset = async () => {
    setActionLoading(true);
    try {
      await resetReplay();
      setPrediction(null);
      await loadPage();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Reset failed.");
    } finally {
      setActionLoading(false);
    }
  };

  const handlePrediction = async () => {
    if (!predictionModel) {
      return;
    }
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

  const kpis = useMemo(() => dashboard?.kpis || {}, [dashboard]);

  if (loading) {
    return <main className="page"><section className="empty-state">Loading dashboard…</section></main>;
  }

  return (
    <main className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">Brazilian Ecommerce Demo</p>
          <h1>Operational replay dashboard</h1>
          <p className="hero-copy">
            Replay immutable Olist order events into the live store, track KPI changes,
            inspect processing health, and review the shipped ML artifacts.
          </p>
        </div>
        <div className="hero-actions">
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

      <section className="grid kpi-grid">
        <KpiCard label="Orders processed" value={kpis.orders ?? 0} />
        <KpiCard label="Revenue" value={currency(kpis.revenue)} />
        <KpiCard label="Delivered orders" value={kpis.delivered_orders ?? 0} />
        <KpiCard label="Cancelled orders" value={kpis.cancelled_orders ?? 0} />
        <KpiCard label="Average order value" value={currency(kpis.avg_order_value)} />
        <KpiCard label="Unique customers" value={kpis.unique_customers ?? 0} />
      </section>

      <section className="grid two-column">
        <Panel title="Replay status" subtitle="Latest orchestration and event-bank totals">
          {replay?.event_bank ? (
            <dl className="meta-list">
              <MetaItem label="Immutable events" value={replay.event_bank.event_count} />
              <MetaItem label="Date range" value={`${replay.event_bank.date_range.start || "n/a"} → ${replay.event_bank.date_range.end || "n/a"}`} />
              <MetaItem label="Batches run" value={replay.recent_batches?.length || 0} />
              <MetaItem label="Recent activities" value={replay.recent_activity?.length || 0} />
            </dl>
          ) : (
            <Empty message="No replay metadata available yet." />
          )}
        </Panel>

        <Panel title="System health" subtitle="Backend connectivity and runtime counts">
          {health ? (
            <dl className="meta-list">
              <MetaItem label="Database mode" value={health.database_url} />
              <MetaItem label="Source data" value={health.source_data_present ? "available" : "missing"} />
              <MetaItem label="Processed events" value={health.database_counts?.processed_events ?? 0} />
              <MetaItem label="Live orders" value={health.database_counts?.orders ?? 0} />
            </dl>
          ) : (
            <Empty message="Health data unavailable." />
          )}
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="Monthly trend" subtitle="Live order and revenue progression">
          <TrendTable rows={dashboard?.trends || []} />
        </Panel>
        <Panel title="Top categories" subtitle="Revenue by translated product category">
          <CategoryTable rows={dashboard?.top_categories || []} />
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="Recent activity" subtitle="Most recent processed events">
          <ActivityTable rows={dashboard?.activity || []} />
        </Panel>
        <Panel title="Recent orders" subtitle="Current live business records">
          <OrdersTable rows={dashboard?.recent_orders || []} />
        </Panel>
      </section>

      <section className="grid two-column">
        <Panel title="ML artifacts" subtitle="Saved models and sample prediction access">
          <div className="ml-controls">
            <select value={predictionModel} onChange={(event) => setPredictionModel(event.target.value)}>
              <option value="">Select a model</option>
              {models
                .filter((model) => model.supports_prediction)
                .map((model) => (
                  <option key={model.model_name} value={model.model_name}>
                    {model.model_name}
                  </option>
                ))}
            </select>
            <button onClick={handlePrediction} disabled={!predictionModel || actionLoading}>
              Run sample prediction
            </button>
          </div>
          <ModelTable rows={models} />
        </Panel>

        <Panel title="Prediction result" subtitle="Inference using the preserved training sample">
          {prediction ? (
            <dl className="meta-list">
              <MetaItem label="Model" value={prediction.model_name} />
              <MetaItem label="Prediction" value={String(prediction.prediction)} />
              <MetaItem label="Observed target" value={String(prediction.target)} />
              <MetaItem label="Features" value={prediction.feature_count} />
              <MetaItem
                label="Score summary"
                value={prediction.score ? JSON.stringify(prediction.score) : "n/a"}
              />
            </dl>
          ) : (
            <Empty message="Select a supported model to inspect a real sample prediction." />
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
          <th>Period</th>
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
    return <Empty message="No category revenue is available yet." />;
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
          <th>Event</th>
          <th>Type</th>
          <th>Order</th>
          <th>Processed</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.event_id}>
            <td>{row.event_id}</td>
            <td>{row.event_type}</td>
            <td>{row.order_id}</td>
            <td>{row.processed_at || "pending"}</td>
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
          <th>Order</th>
          <th>Status</th>
          <th>Customer</th>
          <th>Revenue</th>
          <th>Purchased</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.order_id}>
            <td>{row.order_id}</td>
            <td>{row.status}</td>
            <td>{row.customer_state || "n/a"}</td>
            <td>{currency(row.payment_value || row.revenue)}</td>
            <td>{row.purchase_ts || "n/a"}</td>
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
            <td>{row.model_name}</td>
            <td>{row.supports_prediction ? "available" : "artifact only"}</td>
            <td>{row.feature_count ?? "n/a"}</td>
            <td>{row.score ? JSON.stringify(row.score) : row.error || "n/a"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default App;
