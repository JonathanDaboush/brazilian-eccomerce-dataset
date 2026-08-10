import { useState, useCallback } from "react";
import { validateTrainingCsv, ingestTrainingCsv, getDatasetSchemas } from "./api";

const DATASET_DESCRIPTIONS = {
  delivery_delay: "Order-level features for predicting late deliveries",
  demand_forecasting: "Product/month sales data for demand prediction",
  order_cancellation: "Order-level features for cancellation prediction",
  review_prediction: "Review-level features for review score prediction",
  customer_purchase_prediction: "Customer features for future purchase prediction",
  product_recommendation: "Customer-product purchase counts for recommendations",
};

export default function UploadPage({ toast }) {
  const [dragging, setDragging] = useState(false);
  const [validation, setValidation] = useState(null);
  const [file, setFile] = useState(null);
  const [validating, setValidating] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [result, setResult] = useState(null);
  const [schemas, setSchemas] = useState(null);
  const [showSchemas, setShowSchemas] = useState(false);

  const loadSchemas = async () => {
    if (schemas) { setShowSchemas(s => !s); return; }
    try {
      const res = await getDatasetSchemas();
      setSchemas(res.data);
      setShowSchemas(true);
    } catch {
      toast.error("Failed to load dataset schemas");
    }
  };

  const processFile = useCallback(async (f) => {
    if (!f || !f.name.endsWith(".csv")) {
      toast.error("Only .csv files are supported");
      return;
    }
    setFile(f);
    setValidation(null);
    setResult(null);
    setValidating(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await validateTrainingCsv(fd);
      setValidation(res.data);
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Validation failed");
      setFile(null);
    } finally {
      setValidating(false);
    }
  }, [toast]);

  const handleDrop = useCallback(e => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    processFile(f);
  }, [processFile]);

  const handleFileInput = e => {
    const f = e.target.files[0];
    if (f) processFile(f);
  };

  const handleIngest = async () => {
    if (!file || !validation?.valid) return;
    setIngesting(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await ingestTrainingCsv(fd);
      setResult(res.data);
      toast.success(`✓ ${res.data.rows_added} rows added to ${res.data.dataset}`);
      setFile(null);
      setValidation(null);
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Ingest failed");
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="upload-page">
      <h2>Upload ML Training Data</h2>
      <p className="subtitle">
        Upload a CSV file matching one of the 6 ML training dataset schemas. New rows will be
        appended (deduplicated), and a Kafka event will trigger downstream retraining.
      </p>

      <button className="btn-secondary" onClick={loadSchemas}>
        {showSchemas ? "Hide" : "Show"} Expected Dataset Schemas
      </button>

      {showSchemas && schemas && (
        <div className="schema-grid">
          {Object.entries(schemas).map(([name, cols]) => (
            <div key={name} className="schema-card">
              <strong>{name}</strong>
              <p>{DATASET_DESCRIPTIONS[name]}</p>
              <div className="schema-cols">
                {cols.map(c => <span key={c} className="col-badge">{c}</span>)}
              </div>
            </div>
          ))}
        </div>
      )}

      <div
        className={`drop-zone ${dragging ? "dragging" : ""}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        {validating ? (
          <p>Validating…</p>
        ) : file ? (
          <p>📄 <strong>{file.name}</strong> — {(file.size / 1024).toFixed(1)} KB</p>
        ) : (
          <>
            <p>Drag &amp; drop a <strong>.csv</strong> file here</p>
            <p>or</p>
          </>
        )}
        <label className="file-label">
          Browse
          <input type="file" accept=".csv" hidden onChange={handleFileInput} />
        </label>
      </div>

      {validation && (
        <div className={`validation-panel card ${validation.valid ? "valid" : "invalid"}`}>
          <h3>
            {validation.valid ? "✅ Ready to ingest" : "❌ Validation failed"}
            {" — "}
            <em>{validation.detected_dataset}</em>
          </h3>
          <p>
            <strong>{validation.rows}</strong> rows · <strong>{validation.columns.length}</strong> columns ·
            Existing rows in dataset: <strong>{validation.existing_rows_in_dataset}</strong>
          </p>

          {validation.missing_required_columns.length > 0 && (
            <div className="missing-cols">
              <strong>Missing required columns:</strong>{" "}
              {validation.missing_required_columns.map(c => (
                <span key={c} className="col-badge missing">{c}</span>
              ))}
            </div>
          )}

          <div className="col-validation-grid">
            {validation.column_info.map(c => (
              <div key={c.name} className={`col-item ${c.required ? "required" : ""}`}>
                <span className={`col-status ${c.required ? "green" : "gray"}`}>
                  {c.required ? "✓" : "○"}
                </span>
                <span className="col-name">{c.name}</span>
                <span className="col-dtype">{c.dtype}</span>
                {c.null_count > 0 && <span className="col-nulls">⚠ {c.null_count} nulls</span>}
              </div>
            ))}
          </div>

          <h4>Preview (first 10 rows)</h4>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>{validation.columns.map(c => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {validation.preview.map((row, i) => (
                  <tr key={i}>
                    {validation.columns.map(c => <td key={c}>{String(row[c] ?? "")}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {validation.valid && (
            <button className="btn-primary ingest-btn" disabled={ingesting} onClick={handleIngest}>
              {ingesting ? "Ingesting…" : "Confirm & Ingest"}
            </button>
          )}
        </div>
      )}

      {result && (
        <div className="result-panel card success">
          <h3>✅ Ingest Complete</h3>
          <p>Dataset: <strong>{result.dataset}</strong></p>
          <p>Rows in upload: <strong>{result.rows_in_upload}</strong></p>
          <p>Rows actually added (after dedup): <strong>{result.rows_added}</strong></p>
          <p>Total rows now: <strong>{result.total_rows_now}</strong></p>
          <p>Dedup keys used: <strong>{result.dedup_keys_used?.join(", ") || "none"}</strong></p>
        </div>
      )}
    </div>
  );
}
