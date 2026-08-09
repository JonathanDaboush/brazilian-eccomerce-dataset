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


def check_backend_health(**context) -> dict:
    health = _request_json("/api/health")
    if not health.get("source_data_present"):
        raise RuntimeError("Source data files are missing — cannot proceed with replay.")
    event_count = health.get("event_bank", {}).get("event_count", 0)
    if event_count == 0:
        raise RuntimeError("Event bank is empty — cannot proceed with replay.")
    print(f"Backend healthy. Event bank contains {event_count:,} immutable events.")
    return health


def reset_replay_state(**context) -> dict:
    result = _request_json("/api/replay/reset", method="POST", payload={})
    print("Live state reset. Operational database cleared.")
    return result


def determine_batch(**context) -> dict:
    replay_status = _request_json("/api/replay")
    event_bank = replay_status.get("event_bank", {})
    event_count = event_bank.get("event_count", 0)
    batch_size = min(500, event_count)
    print(f"Determined batch size: {batch_size} events from {event_count:,} total.")
    context["task_instance"].xcom_push(key="batch_size", value=batch_size)
    return {"batch_size": batch_size, "event_count": event_count}


def validate_batch(**context) -> dict:
    replay_status = _request_json("/api/replay")
    event_bank = replay_status.get("event_bank", {})
    event_types = event_bank.get("event_types", {})
    if not event_types:
        raise ValueError("Event bank metadata missing event_type breakdown.")

    required_types = {"new_order", "payment", "delivered"}
    missing = required_types - set(event_types.keys())
    if missing:
        raise ValueError(f"Event bank is missing required event types: {missing}")

    print(f"Batch validated. Event types present: {list(event_types.keys())}")
    print(f"Event type counts: {event_types}")
    return {"event_types": event_types, "validation": "passed"}


def publish_kafka_events(**context) -> dict:
    """Publish a batch of events to Kafka via the backend producer endpoint."""
    batch_size = context["task_instance"].xcom_pull(task_ids="determine_batch", key="batch_size") or 250
    result = _request_json(
        "/api/replay/publish",
        method="POST",
        payload={"start_offset": 0, "limit": batch_size, "pace_ms": 0},
    )
    published = result.get("published_events", 0)
    print(f"Published {published:,} events to Kafka topic '{result.get('topic')}'.")
    return result


def run_direct_replay_batch(**context) -> dict:
    """Replay events directly into the database (fallback / parallel path)."""
    batch_size = context["task_instance"].xcom_pull(task_ids="determine_batch", key="batch_size") or 250
    result = _request_json(
        "/api/replay",
        method="POST",
        payload={"start_offset": 0, "limit": batch_size, "pace_ms": 0},
    )
    processed = result.get("processed_events", 0)
    duplicates = result.get("duplicate_events", 0)
    print(f"Direct replay complete: {processed:,} processed, {duplicates:,} duplicates skipped.")
    return result


def refresh_analytics(**context) -> dict:
    """Fetch the updated dashboard to confirm analytics have been refreshed."""
    dashboard = _request_json("/api/dashboard")
    kpis = dashboard.get("kpis", {})
    print(f"Analytics refreshed. Current KPIs:")
    print(f"  Orders:         {kpis.get('orders', 0):,}")
    print(f"  Revenue:        ${kpis.get('revenue', 0):,.2f}")
    print(f"  Delivered:      {kpis.get('delivered_orders', 0):,}")
    print(f"  Cancelled:      {kpis.get('cancelled_orders', 0):,}")
    print(f"  Avg delivery:   {kpis.get('avg_delivery_days', 'n/a')} days")
    print(f"  Satisfaction:   {kpis.get('avg_satisfaction_score', 'n/a')} / 5")
    return kpis


def verify_dashboard(**context) -> dict:
    dashboard = _request_json("/api/dashboard")
    kpis = dashboard.get("kpis", {})
    if kpis.get("orders", 0) <= 0:
        raise ValueError("Replay did not populate any orders — verify the event bank and consumer.")
    if kpis.get("revenue", 0.0) <= 0.0:
        raise ValueError("Revenue is zero after replay — check payment event processing.")
    print(f"Verification passed. {kpis['orders']:,} orders, ${kpis['revenue']:,.2f} revenue.")
    return kpis


def generate_summary(**context) -> dict:
    """Pull the final dashboard state and emit a human-readable run summary."""
    dashboard = _request_json("/api/dashboard")
    health = _request_json("/api/health")
    kpis = dashboard.get("kpis", {})
    db_counts = health.get("database_counts", {})
    top_categories = dashboard.get("top_categories", [])[:3]

    summary = {
        "run_completed_at": datetime.utcnow().isoformat(),
        "orders_in_database": kpis.get("orders", 0),
        "total_revenue": kpis.get("revenue", 0),
        "delivered_orders": kpis.get("delivered_orders", 0),
        "cancelled_orders": kpis.get("cancelled_orders", 0),
        "avg_delivery_days": kpis.get("avg_delivery_days"),
        "avg_satisfaction_score": kpis.get("avg_satisfaction_score"),
        "processed_events_total": db_counts.get("processed_events", 0),
        "top_categories": [f"{c['category']} (${c['revenue']:,.0f})" for c in top_categories],
    }

    print("=" * 60)
    print("ECOMMERCE REPLAY PIPELINE — RUN SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    return summary


default_args = {
    "owner": "copilot",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="ecommerce_replay_demo",
    description=(
        "End-to-end Olist replay pipeline: health check → batch determination → validation → "
        "Kafka publish → direct DB replay → analytics refresh → verification → summary."
    ),
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["ecommerce", "replay", "demo", "brazilian"],
) as dag:
    health = PythonOperator(
        task_id="check_backend_health",
        python_callable=check_backend_health,
    )
    reset = PythonOperator(
        task_id="reset_replay_state",
        python_callable=reset_replay_state,
    )
    determine = PythonOperator(
        task_id="determine_batch",
        python_callable=determine_batch,
    )
    validate = PythonOperator(
        task_id="validate_batch",
        python_callable=validate_batch,
    )
    publish_kafka = PythonOperator(
        task_id="publish_kafka_events",
        python_callable=publish_kafka_events,
    )
    direct_replay = PythonOperator(
        task_id="run_direct_replay_batch",
        python_callable=run_direct_replay_batch,
    )
    refresh = PythonOperator(
        task_id="refresh_analytics",
        python_callable=refresh_analytics,
    )
    verify = PythonOperator(
        task_id="verify_dashboard",
        python_callable=verify_dashboard,
    )
    summary = PythonOperator(
        task_id="generate_summary",
        python_callable=generate_summary,
    )

    # Pipeline: health → reset → determine → validate → replay paths → refresh → verify → summary
    health >> reset >> determine >> validate >> [publish_kafka, direct_replay] >> refresh >> verify >> summary



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
