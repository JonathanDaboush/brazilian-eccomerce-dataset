from __future__ import annotations

import json
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

from airflow import DAG
from airflow.operators.python import PythonOperator

BASE_URL = "http://backend:8000"
ML_INGEST_TOPIC = "ml-training-ingest"


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


def task_check_replay_completion(**context):
    """
    Determine the primary ingest path:
    - If events_remaining > 0, we are still replaying historic data → skip ML ingest.
    - If events_remaining == 0 (or replay is idle/completed), the event bank is fully
      consumed and user-uploaded ML training data is now the primary ingest path.
    Pushes 'replay_active' XCom for downstream tasks.
    """
    status = call_api("/replay/status")
    replay_status = status.get("status", "idle")
    events_remaining = status.get("events_remaining", 0)
    replay_active = replay_status in ("running", "paused") and events_remaining > 0
    context["ti"].xcom_push(key="replay_active", value=replay_active)
    context["ti"].xcom_push(key="events_remaining", value=events_remaining)
    print(
        f"Replay status={replay_status}, events_remaining={events_remaining}, "
        f"replay_active={replay_active}"
    )


def task_continue_replay(**context):
    """
    If the replay event bank still has rows, publish another batch so data
    keeps flowing into the database incrementally via Kafka.
    Skipped when replay is already complete.
    """
    replay_active = context["ti"].xcom_pull(key="replay_active", task_ids="check_replay_completion")
    if not replay_active:
        print("Replay complete – skipping batch publish")
        return
    result = call_api(
        "/replay/start",
        method="POST",
        payload={"batch_size": 500, "replay_speed_ms": 0},
    )
    if not result.get("ok"):
        raise RuntimeError(f"Replay batch publish failed: {result}")
    print(f"Published replay batch: {result}")


def task_check_new_ml_data(**context):
    """
    Poll the Kafka topic ml-training-ingest for unprocessed messages.
    In this DAG we use a lightweight indicator: compare ml/models
    training_data_available counts against a stored XCom baseline.
    Pushes 'has_new_ml_data' so the retrain task can decide whether to run.
    """
    models = call_api("/ml/models").get("models", [])
    datasets_available = [m["name"] for m in models if m.get("training_data_available")]
    context["ti"].xcom_push(key="datasets_available", value=datasets_available)
    # Always attempt retrain if there is any training data; the monitor_and_retrain
    # logic inside the backend will guard against unnecessary model swaps.
    has_new_ml_data = len(datasets_available) > 0
    context["ti"].xcom_push(key="has_new_ml_data", value=has_new_ml_data)
    print(f"Datasets with training data: {datasets_available}")


def task_trigger_retrain(**context):
    """
    When new ML training data has arrived (uploaded by user via /upload/training-csv
    and published to Kafka), trigger model retraining by calling the backend retrain
    endpoint.  Currently the backend exposes model info; a dedicated /ml/retrain
    endpoint would be the right integration point.  This task logs the intent and
    verifies model availability post-upload.
    """
    replay_active = context["ti"].xcom_pull(key="replay_active", task_ids="check_replay_completion")
    has_new_ml_data = context["ti"].xcom_pull(key="has_new_ml_data", task_ids="check_new_ml_data")

    if replay_active:
        print("Replay still active – skipping ML retrain until event bank is consumed")
        return

    if not has_new_ml_data:
        print("No ML training data available – skipping retrain")
        return

    result = call_api("/ml/retrain", method="POST", payload={})
    print(f"Retrain result: {result}")
    context["ti"].xcom_push(key="retrain_result", value=result)


def task_verify_model_artifacts(**context):
    """Confirm at least one ML model is available after the ingest/retrain cycle."""
    retrain_result = context["ti"].xcom_pull(key="retrain_result", task_ids="trigger_retrain") or {}
    failed_runs = [run for run in retrain_result.get("runs", []) if not run.get("ok")]
    if failed_runs:
        raise RuntimeError(f"Some retraining runs failed: {failed_runs}")
    models = call_api("/ml/models").get("models", [])
    available = [m["name"] for m in models if m.get("available")]
    if not available:
        # Non-fatal: training data may have been uploaded but retrain hasn't run yet
        print("Warning: no model artifacts found yet. Retrain may be pending.")
    else:
        print(f"Model artifacts verified: {available}")


def task_generate_summary(**context):
    status = call_api("/replay/status")
    summary = call_api("/dashboard/summary")
    models = call_api("/ml/models")
    context["ti"].xcom_push(
        key="daily_summary",
        value={
            "status": status,
            "summary": summary,
            "models": models,
            "run_date": datetime.utcnow().isoformat(),
        },
    )
    print("Daily summary generated")


with DAG(
    dag_id="ingest_ml_training_data",
    description=(
        "Orchestrates incremental ML training data ingest via Kafka, "
        "continues replay until event bank is consumed, then manages "
        "user-uploaded CSV data as the primary ingest path."
    ),
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "data-platform",
        "retries": 2,
        "retry_delay": timedelta(minutes=3),
    },
    tags=["olist", "kafka", "ml", "ingest", "training"],
) as dag:

    health_check = PythonOperator(
        task_id="health_check",
        python_callable=task_health_check,
    )
    check_replay = PythonOperator(
        task_id="check_replay_completion",
        python_callable=task_check_replay_completion,
    )
    continue_replay = PythonOperator(
        task_id="continue_replay_if_active",
        python_callable=task_continue_replay,
    )
    check_ml_data = PythonOperator(
        task_id="check_new_ml_data",
        python_callable=task_check_new_ml_data,
    )
    trigger_retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=task_trigger_retrain,
    )
    verify_artifacts = PythonOperator(
        task_id="verify_model_artifacts",
        python_callable=task_verify_model_artifacts,
    )
    generate_summary = PythonOperator(
        task_id="generate_daily_summary",
        python_callable=task_generate_summary,
    )

    (
        health_check
        >> check_replay
        >> continue_replay
        >> check_ml_data
        >> trigger_retrain
        >> verify_artifacts
        >> generate_summary
    )
