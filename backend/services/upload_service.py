from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from kafka import KafkaProducer

logger = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parents[1] / "ml_training_data"
TRAINING_VERSIONS_DIR = Path(__file__).resolve().parents[1] / "training_dataset_versions"
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
ML_INGEST_TOPIC = "ml-training-ingest"

DATASET_SIGNATURES: dict[str, frozenset[str]] = {
    "delivery_delay": frozenset({"order_id", "customer_id", "late_delivery"}),
    "demand_forecasting": frozenset({"product_id", "month", "units_sold"}),
    "order_cancellation": frozenset({"order_id", "customer_id", "cancelled"}),
    "review_prediction": frozenset({"review_id", "order_id", "review_score"}),
    "customer_purchase_prediction": frozenset({"customer_id", "customer_unique_id", "future_purchase"}),
    "product_recommendation": frozenset({"customer_id", "product_id", "purchase_count"}),
}

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


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc


def _dataset_root(dataset_name: str) -> Path:
    return TRAINING_VERSIONS_DIR / dataset_name


def _latest_pointer_path(dataset_name: str) -> Path:
    return _dataset_root(dataset_name) / "latest.json"


def get_dataset_file_path(dataset_name: str) -> Path:
    latest_pointer = _latest_pointer_path(dataset_name)
    if latest_pointer.exists():
        metadata = json.loads(latest_pointer.read_text())
        merged_path = Path(metadata["merged_dataset_path"])
        if merged_path.exists():
            return merged_path
    return TRAINING_DIR / f"{dataset_name}.csv"


def get_dataset_version_info(dataset_name: str) -> dict[str, Any]:
    active_path = get_dataset_file_path(dataset_name)
    latest_pointer = _latest_pointer_path(dataset_name)
    if latest_pointer.exists():
        metadata = json.loads(latest_pointer.read_text())
        metadata["active_dataset_path"] = str(active_path)
        return metadata
    return {
        "dataset": dataset_name,
        "version_id": None,
        "active_dataset_path": str(active_path),
        "base_dataset_path": str(TRAINING_DIR / f"{dataset_name}.csv"),
    }


def _validation_payload(df: pd.DataFrame, filename: str, dataset_name: str) -> dict[str, Any]:
    required_cols = sorted(DATASET_SIGNATURES[dataset_name])
    dedup_keys = DEDUP_KEYS.get(dataset_name, [])
    missing_cols = [c for c in required_cols if c not in df.columns]

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    preview = df.head(10).fillna("").to_dict(orient="records")
    duplicate_rows = int(df.duplicated().sum())
    duplicate_keys = 0
    if dedup_keys and all(key in df.columns for key in dedup_keys):
        duplicate_keys = int(df.duplicated(subset=dedup_keys).sum())

    active_path = get_dataset_file_path(dataset_name)
    existing_rows = 0
    if active_path.exists():
        existing_rows = max(sum(1 for _ in active_path.open()) - 1, 0)

    column_info = []
    for col in df.columns:
        column_info.append(
            {
                "name": col,
                "required": col in DATASET_SIGNATURES[dataset_name],
                "null_count": null_counts[col],
                "dtype": str(df[col].dtype),
                "example": "" if df[col].dropna().empty else str(df[col].dropna().iloc[0]),
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
        "column_info": column_info,
        "preview": preview,
        "duplicate_rows": duplicate_rows,
        "duplicate_business_keys": duplicate_keys,
        "dedup_keys": dedup_keys,
        "existing_rows_in_dataset": existing_rows,
        "active_dataset_path": str(active_path),
        "active_dataset_version": get_dataset_version_info(dataset_name),
    }


def validate_training_csv(content: bytes, filename: str) -> dict[str, Any]:
    df = _read_csv_bytes(content)
    dataset_name = detect_dataset(list(df.columns))
    if dataset_name is None:
        raise ValueError(
            "Cannot identify dataset from column names. Expected columns matching one of: "
            + ", ".join(f"{k}: {sorted(v)}" for k, v in DATASET_SIGNATURES.items())
        )
    return _validation_payload(df, filename, dataset_name)


def ingest_training_csv(content: bytes, filename: str) -> dict[str, Any]:
    validation = validate_training_csv(content, filename)
    if not validation["valid"]:
        raise ValueError(f"Missing required columns: {validation['missing_required_columns']}")

    dataset_name = validation["detected_dataset"]
    df = _read_csv_bytes(content)
    current_dataset_path = get_dataset_file_path(dataset_name)
    dedup_keys = DEDUP_KEYS.get(dataset_name, [])

    if current_dataset_path.exists():
        current_df = pd.read_csv(current_dataset_path)
        rows_before = len(current_df)
    else:
        current_df = pd.DataFrame(columns=df.columns)
        rows_before = 0

    merged = pd.concat([current_df, df], ignore_index=True)
    if dedup_keys and all(key in merged.columns for key in dedup_keys):
        merged = merged.drop_duplicates(subset=dedup_keys, keep="last")
    else:
        merged = merged.drop_duplicates()

    merged = merged.reset_index(drop=True)
    rows_added = max(len(merged) - rows_before, 0)

    version_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    version_root = _dataset_root(dataset_name) / version_id
    version_root.mkdir(parents=True, exist_ok=True)

    uploaded_copy_path = version_root / filename
    uploaded_copy_path.write_bytes(content)

    merged_dataset_path = version_root / f"{dataset_name}.csv"
    merged.to_csv(merged_dataset_path, index=False)

    metadata = {
        "dataset": dataset_name,
        "version_id": version_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": filename,
        "uploaded_copy_path": str(uploaded_copy_path),
        "merged_dataset_path": str(merged_dataset_path),
        "base_dataset_path": str(TRAINING_DIR / f"{dataset_name}.csv"),
        "parent_dataset_path": str(current_dataset_path),
        "rows_in_upload": int(len(df)),
        "rows_before": int(rows_before),
        "rows_after": int(len(merged)),
        "rows_added": int(rows_added),
        "dedup_keys": dedup_keys,
    }
    (version_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _latest_pointer_path(dataset_name).parent.mkdir(parents=True, exist_ok=True)
    _latest_pointer_path(dataset_name).write_text(json.dumps(metadata, indent=2))

    _publish_to_kafka(dataset_name, df, rows_added, version_id, str(merged_dataset_path))

    return {
        "ok": True,
        "dataset": dataset_name,
        "version_id": version_id,
        "rows_in_upload": int(len(df)),
        "rows_added": int(rows_added),
        "total_rows_now": int(len(merged)),
        "dedup_keys_used": dedup_keys,
        "uploaded_copy_path": str(uploaded_copy_path),
        "merged_dataset_path": str(merged_dataset_path),
    }


def _publish_to_kafka(
    dataset_name: str,
    df: pd.DataFrame,
    rows_added: int,
    version_id: str,
    merged_dataset_path: str,
) -> None:
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
            "dataset_version_id": version_id,
            "rows_added": rows_added,
            "merged_dataset_path": merged_dataset_path,
            "sample_row": df.head(1).fillna("").to_dict(orient="records")[0] if len(df) > 0 else {},
        }
        producer.send(ML_INGEST_TOPIC, key=dataset_name, value=event)
        producer.flush()
        producer.close()
        logger.info(
            "Published ml_training_ingest event for dataset=%s version=%s rows_added=%d",
            dataset_name,
            version_id,
            rows_added,
        )
    except Exception as exc:
        logger.warning("Failed to publish Kafka event for ML ingest: %s", exc)
