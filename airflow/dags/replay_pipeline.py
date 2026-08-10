from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

from airflow import DAG
from airflow.operators.python import PythonOperator

BASE_URL = "http://backend:8000"


def call_api(path: str, method: str = "GET", payload: dict | None = None):
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=60) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def task_health_check(**_):
    result = call_api("/health/system")
    if result.get("database") != "Healthy":
        raise RuntimeError(f"Database is not healthy: {result}")


def task_build_event_bank(**_):
    call_api("/replay/event-bank/build", method="POST")


def task_publish_batch(**_):
    result = call_api("/replay/start", method="POST", payload={"batch_size": 250, "replay_speed_ms": 5})
    if not result.get("ok"):
        raise RuntimeError(f"Replay start failed: {result}")
    return result


def task_verify_consumer(**context):
    published = context["ti"].xcom_pull(task_ids="publish_kafka_events")
    expected = int((published or {}).get("produced", 0))
    baseline = int((published or {}).get("state", {}).get("events_processed", 0))
    deadline = time.time() + 90
    latest = call_api("/replay/status")
    while time.time() < deadline:
        latest = call_api("/replay/status")
        if int(latest.get("events_processed", 0)) >= baseline + expected:
            return latest
        if latest.get("status") == "failed":
            raise RuntimeError(f"Replay failed while waiting for consumer: {latest}")
        time.sleep(5)
    raise RuntimeError(f"Consumer did not process published events in time: {latest}")


def task_update_metrics(**_):
    summary = call_api("/dashboard/summary")
    if "kpis" not in summary or "processing_status" not in summary:
        raise RuntimeError("Missing KPI payload")


def task_generate_summary(**context):
    status = call_api("/replay/status")
    summary = call_api("/dashboard/summary")
    context["ti"].xcom_push(key="daily_summary", value={"status": status, "summary": summary})


def task_record_completion(**_):
    # state timestamp is already persisted by backend replay_state updates.
    call_api("/health/system")


with DAG(
    dag_id="olist_replay_orchestration",
    description="Orchestrates Olist replay, Kafka processing, and metric updates",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "data-platform",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["olist", "kafka", "replay", "ml"],
) as dag:
    health_check = PythonOperator(task_id="health_check", python_callable=task_health_check)
    build_event_bank = PythonOperator(task_id="build_event_bank", python_callable=task_build_event_bank)
    publish_batch = PythonOperator(task_id="publish_kafka_events", python_callable=task_publish_batch)
    verify_consumer = PythonOperator(task_id="verify_consumer_processing", python_callable=task_verify_consumer)
    update_metrics = PythonOperator(task_id="update_business_metrics", python_callable=task_update_metrics)
    generate_summary = PythonOperator(task_id="generate_daily_summary", python_callable=task_generate_summary)
    record_completion = PythonOperator(task_id="record_completion", python_callable=task_record_completion)

    health_check >> build_event_bank >> publish_batch >> verify_consumer >> update_metrics >> generate_summary >> record_completion
