from __future__ import annotations

import json
from datetime import datetime, timedelta
from urllib import request

from airflow import DAG
from airflow.operators.python import PythonOperator


API_BASE = "http://backend:8000"


def _request_json(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(f"{API_BASE}{path}", data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def check_backend_health() -> dict:
    return _request_json("/api/health")


def reset_replay_state() -> dict:
    return _request_json("/api/replay/reset", method="POST", payload={})


def run_replay_batch() -> dict:
    return _request_json(
        "/api/replay",
        method="POST",
        payload={"start_offset": 0, "limit": 250, "pace_ms": 0},
    )


def verify_dashboard() -> dict:
    dashboard = _request_json("/api/dashboard")
    if dashboard["kpis"]["orders"] <= 0:
        raise ValueError("Replay did not populate any orders.")
    return dashboard["kpis"]


default_args = {
    "owner": "copilot",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="ecommerce_replay_demo",
    description="Reset replay state, load a real Olist event batch, and verify business KPIs.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["ecommerce", "replay", "demo"],
) as dag:
    health = PythonOperator(task_id="check_backend_health", python_callable=check_backend_health)
    reset = PythonOperator(task_id="reset_replay_state", python_callable=reset_replay_state)
    replay = PythonOperator(task_id="run_replay_batch", python_callable=run_replay_batch)
    verify = PythonOperator(task_id="verify_dashboard", python_callable=verify_dashboard)

    health >> reset >> replay >> verify
