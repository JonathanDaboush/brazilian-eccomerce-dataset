from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from kafka import KafkaProducer

logger = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parents[1] / "ml_training_data"
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
ML_INGEST_TOPIC = "ml-training-ingest"

# Column signatures used to auto-detect which dataset a CSV belongs to.
# Keys are dataset names; values are frozensets of required column names.
DATASET_SIGNATURES: dict[str, frozenset[str]] = {
    "delivery_delay": frozenset({"order_id", "customer_id", "late_delivery"}),
    "demand_forecasting": frozenset({"product_id", "month", "units_sold"}),
    "order_cancellation": frozenset({"order_id", "customer_id", "cancelled"}),
    "review_prediction": frozenset({"review_id", "order_id", "review_score"}),
    "customer_purchase_prediction": frozenset(
        {"customer_id", "customer_unique_id", "future_purchase"}
    ),
    "product_recommendation": frozenset({"customer_id", "product_id", "purchase_count"}),
}

# Natural dedup keys per dataset (used when appending to avoid duplicates)
DEDUP_KEYS: dict[str, list[str]] = {
    "delivery_delay": ["order_id"],
    "demand_forecasting": ["product_id", "month"],
    "order_cancellation": ["order_id"],
    "review_prediction": ["review_id"],
    "customer_purchase_prediction": ["customer_id"],
    "product_recommendation": ["customer_id", "product_id"],
}


def detect_dataset(columns: list[str]) -> str | None:
    col_set = frozenset(columns)
    for name, required in DATASET_SIGNATURES.items():
        if required.issubset(col_set):
            return name
    return None


def validate_training_csv(content: bytes, filename: str) -> dict[str, Any]:
    """Parse and validate a CSV without writing anything to disk."""
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    dataset_name = detect_dataset(list(df.columns))
    if dataset_name is None:
        raise ValueError(
            "Cannot identify dataset from column names. "
            "Expected columns matching one of: "
            + ", ".join(f"{k}: {sorted(v)}" for k, v in DATASET_SIGNATURES.items())
        )

    required_cols = sorted(DATASET_SIGNATURES[dataset_name])
    present = set(df.columns)
    missing_cols = [c for c in required_cols if c not in present]

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    preview = df.head(10).fillna("").to_dict(orient="records")

    existing_path = TRAINING_DIR / f"{dataset_name}.csv"
    existing_rows = 0
    if existing_path.exists():
        existing_rows = sum(1 for _ in open(existing_path)) - 1  # subtract header

    # Column-level validation info
    col_info = []
    for col in df.columns:
        col_info.append(
            {
                "name": col,
                "required": col in DATASET_SIGNATURES[dataset_name],
                "null_count": null_counts[col],
                "dtype": str(df[col].dtype),
            }
        )

    return {
        "filename": filename,
        "detected_dataset": dataset_name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing_cols,
        "valid": len(missing_cols) == 0,
        "null_counts": null_counts,
        "column_info": col_info,
        "preview": preview,
        "existing_rows_in_dataset": existing_rows,
    }


def ingest_training_csv(content: bytes, filename: str) -> dict[str, Any]:
    """Append valid rows to the training CSV and publish Kafka events."""
    validation = validate_training_csv(content, filename)
    if not validation["valid"]:
        raise ValueError(
            f"Missing required columns: {validation['missing_required_columns']}"
        )

    dataset_name = validation["detected_dataset"]
    df = pd.read_csv(pd.io.common.BytesIO(content))

    existing_path = TRAINING_DIR / f"{dataset_name}.csv"
    dedup_keys = DEDUP_KEYS.get(dataset_name, [])

    rows_before = 0
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        rows_before = len(existing_df)
        if dedup_keys and all(k in df.columns and k in existing_df.columns for k in dedup_keys):
            merged = pd.concat([existing_df, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=dedup_keys, keep="last")
        else:
            merged = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
    else:
        merged = df
        rows_before = 0

    merged.to_csv(existing_path, index=False)
    rows_added = len(merged) - rows_before

    # Publish each new row to Kafka so the consumer/Airflow can react
    _publish_to_kafka(dataset_name, df, rows_added)

    return {
        "ok": True,
        "dataset": dataset_name,
        "rows_in_upload": int(len(df)),
        "rows_added": rows_added,
        "total_rows_now": int(len(merged)),
        "dedup_keys_used": dedup_keys,
    }


def _publish_to_kafka(dataset_name: str, df: pd.DataFrame, rows_added: int) -> None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            key_serializer=lambda x: x.encode("utf-8"),
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            retries=3,
            request_timeout_ms=5000,
        )
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "ml_training_ingest",
            "dataset": dataset_name,
            "rows_added": rows_added,
            "sample_row": df.head(1).fillna("").to_dict(orient="records")[0] if len(df) > 0 else {},
        }
        producer.send(ML_INGEST_TOPIC, key=dataset_name, value=event)
        producer.flush()
        producer.close()
        logger.info("Published ml_training_ingest event for dataset=%s rows_added=%d", dataset_name, rows_added)
    except Exception as exc:
        # Non-fatal: CSV has already been written; log and continue
        logger.warning("Failed to publish Kafka event for ML ingest: %s", exc)
