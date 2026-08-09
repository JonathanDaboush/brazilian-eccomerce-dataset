from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_session, init_db
from controller.producer import publish_events
from services import (
    ensure_event_bank,
    get_batch_history,
    get_dashboard_snapshot,
    get_recent_activity,
    get_system_health,
    list_ml_models,
    load_event_bank,
    predict_from_training_sample,
    reset_replay_state,
    run_direct_replay,
)


app = FastAPI(
    title="Brazilian Ecommerce Replay API",
    version="1.0.0",
    description="Replay immutable Olist order events into an operational store and expose business KPIs.",
)


def _allowed_origins() -> list[str]:
    explicit = os.getenv("CORS_ORIGINS")
    if explicit:
        return [origin.strip() for origin in explicit.split(",") if origin.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReplayRequest(BaseModel):
    start_offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=200, ge=1)
    pace_ms: int = Field(default=0, ge=0, le=5000)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    ensure_event_bank()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Brazilian ecommerce replay API",
        "docs": "/docs",
        "health": "/api/health",
        "dashboard": "/api/dashboard",
    }


@app.get("/api/health")
def health(session: Session = Depends(get_session)) -> dict:
    return get_system_health(session)


@app.get("/api/dashboard")
def dashboard(session: Session = Depends(get_session)) -> dict:
    return get_dashboard_snapshot(session)


@app.get("/api/replay")
def replay_status(session: Session = Depends(get_session)) -> dict:
    return {
        "event_bank": ensure_event_bank(),
        "recent_batches": get_batch_history(session),
        "recent_activity": get_recent_activity(session),
    }


@app.post("/api/replay")
def replay_events(request: ReplayRequest, session: Session = Depends(get_session)) -> dict:
    try:
        return run_direct_replay(
            session,
            start_offset=request.start_offset,
            limit=request.limit,
            pace_ms=request.pace_ms,
        )
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/replay/publish")
def publish_replay_events(request: ReplayRequest) -> dict:
    try:
        return publish_events(
            start_offset=request.start_offset,
            limit=request.limit,
            pace_ms=request.pace_ms,
        )
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/replay/reset")
def reset_replay(session: Session = Depends(get_session)) -> dict[str, str]:
    reset_replay_state(session)
    return {"status": "reset"}


@app.get("/api/events")
def event_bank_preview(
    start_offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict:
    events = load_event_bank()
    return {
        "metadata": ensure_event_bank(),
        "events": events[start_offset : start_offset + limit],
    }


@app.get("/api/activity")
def activity(limit: int = Query(default=20, ge=1, le=100), session: Session = Depends(get_session)) -> dict:
    return {"activity": get_recent_activity(session, limit=limit)}


@app.get("/api/ml/models")
def ml_models() -> dict:
    return {"models": list_ml_models()}


@app.get("/api/ml/predict/{model_name}")
def ml_prediction(model_name: str, row_index: int = Query(default=0, ge=0)) -> dict:
    try:
        return predict_from_training_sample(model_name, row_index=row_index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise HTTPException(status_code=500, detail=str(exc)) from exc
