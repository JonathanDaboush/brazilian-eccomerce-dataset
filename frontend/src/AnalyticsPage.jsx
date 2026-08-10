import { useState, useEffect } from "react";
import { getCustomerFeatures, getSellerFeatures, getProductFeatures } from "./api";

const FEATURE_TYPES = [
  { key: "customer", label: "Customer Features", fetch: getCustomerFeatures },
  { key: "seller", label: "Seller Features", fetch: getSellerFeatures },
  { key: "product", label: "Product Features", fetch: getProductFeatures },
];

function FeatureTable({ items }) {
  const [filter, setFilter] = useState("");
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState("asc");

  if (!items || items.length === 0) return <p>No data yet. Run some replay batches first.</p>;

  const cols = Object.keys(items[0]);

  const handleSort = col => {
    if (sortCol === col) {
      setSortDir(d => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortCol(col);
      setSortDir("asc");
    }
  };

  let rows = items;
  if (filter) {
    const q = filter.toLowerCase();
    rows = rows.filter(r => Object.values(r).some(v => String(v ?? "").toLowerCase().includes(q)));
  }
  if (sortCol) {
    rows = [...rows].sort((a, b) => {
      const av = a[sortCol] ?? "";
      const bv = b[sortCol] ?? "";
      return sortDir === "asc"
        ? String(av).localeCompare(String(bv), undefined, { numeric: true })
        : String(bv).localeCompare(String(av), undefined, { numeric: true });
    });
  }

  return (
    <>
      <input
        className="filter-input"
        placeholder="Filter rows…"
        value={filter}
        onChange={e => setFilter(e.target.value)}
      />
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {cols.map(c => (
                <th key={c} className="sortable" onClick={() => handleSort(c)}>
                  {c}
                  {sortCol === c ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {cols.map(c => <td key={c}>{String(row[c] ?? "")}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function AnalyticsPage({ toast }) {
  const [activeType, setActiveType] = useState("customer");
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(false);

  const load = async key => {
    setActiveType(key);
    if (data[key]) return;
    setLoading(true);
    try {
      const ft = FEATURE_TYPES.find(f => f.key === key);
      const res = await ft.fetch(100);
      setData(prev => ({ ...prev, [key]: res.data.items || [] }));
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Failed to load features");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load("customer"); }, []);

  const reload = async () => {
    setData({});
    load(activeType);
  };

  return (
    <div>
      <h2>Analytics Explorer</h2>
      <p className="subtitle">
        Explore engineered feature tables built from replayed order data. Click a column header to sort.
      </p>
      <div className="tab-bar">
        {FEATURE_TYPES.map(ft => (
          <button
            key={ft.key}
            className={`tab-btn ${activeType === ft.key ? "active" : ""}`}
            onClick={() => load(ft.key)}
          >
            {ft.label}
          </button>
        ))}
        <button className="btn-secondary" onClick={reload} style={{ marginLeft: "auto" }}>
          ↺ Refresh
        </button>
      </div>
      {loading ? (
        <p>Loading…</p>
      ) : (
        <FeatureTable items={data[activeType]} />
      )}
    </div>
  );
}
