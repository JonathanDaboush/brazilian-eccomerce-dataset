from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from controller.producer import publish_next_batch
from database import engine, test_connection
from services.ml_service import available_models, predict
from services.replay_service import (
    append_log,
    build_event_bank_if_missing,
    ensure_replay_schema,
    get_dashboard_summary,
    get_replay_state,
    get_trends,
    update_replay_state,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("olist-api")

app = FastAPI(title="Olist Replay API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReplayStartRequest(BaseModel):
    batch_size: int = Field(default=200, ge=1, le=5000)
    replay_speed_ms: int = Field(default=0, ge=0, le=10000)


class PredictRequest(BaseModel):
    features: dict[str, Any] | None = None


@app.on_event("startup")
def startup() -> None:
    ensure_replay_schema()
    test_connection()


@app.get("/health/system")
def health_system():
    db_ok = True
    db_error = None
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        db_ok = False
        db_error = str(exc)

    replay_state = get_replay_state()
    airflow_state = "configured" if True else "unknown"
    kafka_state = "configured"

    status = "Healthy" if db_ok else "Failed"
    if replay_state["status"] in {"running"}:
        status = "Processing"
    if replay_state["status"] in {"failed"}:
        status = "Failed"

    return {
        "status": status,
        "api": "Healthy",
        "database": "Healthy" if db_ok else "Failed",
        "database_error": db_error,
        "kafka": kafka_state,
        "airflow": airflow_state,
        "replay": replay_state,
        "checked_at": datetime.utcnow().isoformat(),
    }


@app.post("/replay/event-bank/build")
def build_event_bank():
    try:
        result = build_event_bank_if_missing()
        return {"ok": True, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/replay/status")
def replay_status():
    return get_replay_state()


@app.post("/replay/start")
def replay_start(request: ReplayStartRequest):
    try:
        build_event_bank_if_missing()
        update_replay_state(status="running", batch_size=request.batch_size, replay_speed_ms=request.replay_speed_ms)
        result = publish_next_batch(batch_size=request.batch_size, replay_speed_ms=request.replay_speed_ms)
        return {"ok": True, **result, "state": get_replay_state()}
    except Exception as exc:
        update_replay_state(status="failed", last_error=str(exc))
        append_log("failed", f"Producer failure: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/replay/pause")
def replay_pause():
    update_replay_state(status="paused")
    return {"ok": True, "state": get_replay_state()}


@app.post("/replay/stop")
def replay_stop():
    update_replay_state(status="stopped")
    return {"ok": True, "state": get_replay_state()}


@app.post("/replay/reset")
def replay_reset():
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE processed_events"))
        conn.execute(text("TRUNCATE TABLE replay_orders"))
        conn.execute(text("TRUNCATE TABLE replay_order_items"))
        conn.execute(text("TRUNCATE TABLE replay_payments"))
        conn.execute(text("TRUNCATE TABLE replay_reviews"))
        conn.execute(text("TRUNCATE TABLE consumer_events_log"))
        conn.execute(text("TRUNCATE TABLE replay_batches"))
        conn.execute(text("TRUNCATE TABLE customer_features"))
        conn.execute(text("TRUNCATE TABLE seller_features"))
        conn.execute(text("TRUNCATE TABLE product_features"))
    update_replay_state(status="idle", current_offset=0, last_error=None, last_batch_produced=0)
    return {"ok": True, "state": get_replay_state()}


@app.get("/dashboard/summary")
def dashboard_summary():
    return get_dashboard_summary()


@app.get("/dashboard/trends")
def dashboard_trends():
    return get_trends()


@app.get("/features/customer")
def customer_features(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM customer_features ORDER BY updated_at DESC LIMIT :limit"),
            {"limit": limit},
        ).mappings().all()
    return {"items": [dict(r) for r in rows]}


@app.get("/features/seller")
def seller_features(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM seller_features ORDER BY updated_at DESC LIMIT :limit"),
            {"limit": limit},
        ).mappings().all()
    return {"items": [dict(r) for r in rows]}


@app.get("/features/product")
def product_features(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM product_features ORDER BY updated_at DESC LIMIT :limit"),
            {"limit": limit},
        ).mappings().all()
    return {"items": [dict(r) for r in rows]}


@app.get("/ml/models")
def ml_models():
    return {"models": available_models()}


@app.post("/ml/predict/{model_name}")
def ml_predict(model_name: str, request: PredictRequest):
    try:
        return predict(model_name, request.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


REQUIRED_UPLOAD_COLUMNS = {
    "delivery_delay": {"late_delivery"},
    "order_cancellation": {"cancelled"},
    "review_prediction": {"review_score"},
    "demand_forecasting": {"units_sold"},
}


@app.post("/excel/validate")
async def validate_excel(file: UploadFile):
    filename = file.filename or ""
    lowered = filename.lower()
    if not (lowered.endswith(".xlsx") or lowered.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx and .xls files are supported")

    import pandas as pd

    content = await file.read()
    try:
        df = pd.read_excel(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {exc}") from exc

    preview = df.head(10).fillna("").to_dict(orient="records")
    checks = {}
    for dataset, required_cols in REQUIRED_UPLOAD_COLUMNS.items():
        missing = sorted(list(required_cols - set(df.columns)))
        checks[dataset] = {"valid": len(missing) == 0, "missing_columns": missing}

    null_counts = df.isna().sum().to_dict()

    return {
        "filename": filename,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": null_counts,
        "validation": checks,
        "preview": preview,
    }
