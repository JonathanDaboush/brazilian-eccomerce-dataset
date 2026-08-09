from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from constants import PREPROCESS_CONFIG, PRIMARY_TARGET
from database import (
    CategoryTranslation,
    Customer,
    Geolocation,
    Order,
    OrderItem,
    Payment,
    ProcessedEvent,
    Product,
    ReplayBatch,
    Review,
    Seller,
)
from ml_functions.pre_process import preprocess_data


BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "original_data"
EVENT_BANK_DIR = BACKEND_DIR / "runtime" / "event_bank"
EVENT_BANK_PATH = EVENT_BANK_DIR / "olist_order_events.jsonl"
EVENT_BANK_METADATA_PATH = EVENT_BANK_DIR / "olist_order_events.metadata.json"
MODEL_DIR = BACKEND_DIR / "models"
ML_DATA_DIR = BACKEND_DIR / "ml_training_data"

EVENT_BANK_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = {
    "customers": "customers.csv",
    "sellers": "sellers.csv",
    "products": "products.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "order_payments.csv",
    "reviews": "order_reviews.csv",
    "geolocation": "geolocation.csv",
    "category_translation": "category_trans.csv",
}

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def _clean_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_value(value) for key, value in record.items()}


def _to_datetime(value: Any) -> datetime | None:
    if value in (None, "", "NaT"):
        return None
    if isinstance(value, datetime):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _upsert(instance_cls, session: Session, payload: dict[str, Any], pk_name: str):
    payload = {key: value for key, value in payload.items() if key in instance_cls.__table__.columns}
    primary_value = payload.get(pk_name)
    if primary_value in (None, ""):
        return None

    existing = session.get(instance_cls, primary_value)
    if existing is None:
        existing = instance_cls(**{pk_name: primary_value})
        session.add(existing)

    for key, value in payload.items():
        setattr(existing, key, value)
    return existing


@lru_cache(maxsize=1)
def load_source_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in SOURCE_FILES.items():
        path = DATA_DIR / filename
        parse_dates = DATE_COLUMNS.get(name)
        tables[name] = pd.read_csv(path, parse_dates=parse_dates)
    return tables


def _build_order_event_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tables = load_source_tables()
    orders = tables["orders"].sort_values("order_purchase_timestamp")
    customers = tables["customers"].set_index("customer_id")
    products = tables["products"].set_index("product_id")
    sellers = tables["sellers"].set_index("seller_id")
    categories = tables["category_translation"].set_index("product_category_name")

    items_by_order = {key: frame for key, frame in tables["order_items"].groupby("order_id", sort=False)}
    payments_by_order = {key: frame for key, frame in tables["payments"].groupby("order_id", sort=False)}
    reviews_by_order = {key: frame for key, frame in tables["reviews"].groupby("order_id", sort=False)}

    events: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for _, order_row in orders.iterrows():
        order = _clean_record(order_row.to_dict())
        order_id = order["order_id"]
        customer = None
        if order["customer_id"] in customers.index:
            customer = _clean_record(customers.loc[order["customer_id"]].to_dict())

        raw_items = items_by_order.get(order_id, pd.DataFrame())
        item_records = []
        seller_records: dict[str, dict[str, Any]] = {}
        product_records: dict[str, dict[str, Any]] = {}
        category_records: dict[str, dict[str, Any]] = {}

        if not raw_items.empty:
            for _, item_row in raw_items.iterrows():
                item = _clean_record(item_row.to_dict())
                item_records.append(item)

                product_id = item.get("product_id")
                if product_id in products.index:
                    product = _clean_record(products.loc[product_id].to_dict())
                    product_records[product_id] = product
                    category_name = product.get("product_category_name")
                    if category_name in categories.index:
                        category_records[category_name] = _clean_record(categories.loc[category_name].to_dict())

                seller_id = item.get("seller_id")
                if seller_id in sellers.index:
                    seller_records[seller_id] = _clean_record(sellers.loc[seller_id].to_dict())

        payment_records = []
        if order_id in payments_by_order:
            payment_records = [
                _clean_record(record)
                for record in payments_by_order[order_id].to_dict(orient="records")
            ]

        review_records = []
        if order_id in reviews_by_order:
            review_records = [
                _clean_record(record)
                for record in reviews_by_order[order_id].to_dict(orient="records")
            ]

        base_payload = {
            "order": order,
            "customer": customer,
            "items": item_records,
            "products": list(product_records.values()),
            "sellers": list(seller_records.values()),
            "category_translation": list(category_records.values()),
        }

        def add_event(event_type: str, sequence: int, event_time: Any, payload: dict[str, Any], order_state: dict[str, Any]):
            if _to_datetime(event_time) is None:
                return
            event = {
                "event_id": f"{order_id}:{sequence}:{event_type}",
                "order_id": order_id,
                "event_type": event_type,
                "event_time": _clean_value(event_time),
                "sequence": sequence,
                "order_state": _clean_record(order_state),
                "payload": payload,
            }
            status_counts[event_type] += 1
            events.append(event)

        add_event(
            "new_order",
            1,
            order.get("order_purchase_timestamp"),
            base_payload,
            {
                "customer_id": order.get("customer_id"),
                "order_status": "created",
                "order_purchase_timestamp": order.get("order_purchase_timestamp"),
                "order_estimated_delivery_date": order.get("order_estimated_delivery_date"),
            },
        )

        if payment_records:
            add_event(
                "payment",
                2,
                order.get("order_approved_at") or order.get("order_purchase_timestamp"),
                {"payments": payment_records},
                {
                    "order_status": "approved",
                    "order_approved_at": order.get("order_approved_at") or order.get("order_purchase_timestamp"),
                },
            )

        add_event(
            "shipped",
            3,
            order.get("order_delivered_carrier_date"),
            {},
            {
                "order_status": "shipped",
                "order_delivered_carrier_date": order.get("order_delivered_carrier_date"),
            },
        )

        add_event(
            "delivered",
            4,
            order.get("order_delivered_customer_date"),
            {"reviews": review_records},
            {
                "order_status": "delivered",
                "order_delivered_customer_date": order.get("order_delivered_customer_date"),
            },
        )

        if str(order.get("order_status", "")).lower() == "canceled":
            add_event(
                "cancelled",
                5,
                order.get("order_approved_at") or order.get("order_purchase_timestamp"),
                {"reviews": review_records},
                {
                    "order_status": "cancelled",
                    "order_approved_at": order.get("order_approved_at") or order.get("order_purchase_timestamp"),
                },
            )

    events.sort(key=lambda event: (event["event_time"] or "", event["event_id"]))
    metadata = {
        "generated_at": datetime.utcnow().isoformat(),
        "event_count": len(events),
        "date_range": {
            "start": events[0]["event_time"] if events else None,
            "end": events[-1]["event_time"] if events else None,
        },
        "event_types": dict(status_counts),
        "source_files": SOURCE_FILES,
    }
    return events, metadata


def ensure_event_bank(force_rebuild: bool = False) -> dict[str, Any]:
    if force_rebuild or not EVENT_BANK_PATH.exists() or not EVENT_BANK_METADATA_PATH.exists():
        events, metadata = _build_order_event_rows()
        with EVENT_BANK_PATH.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        EVENT_BANK_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return json.loads(EVENT_BANK_METADATA_PATH.read_text(encoding="utf-8"))


def load_event_bank() -> list[dict[str, Any]]:
    ensure_event_bank()
    with EVENT_BANK_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _batch_window(events: list[dict[str, Any]], start_offset: int, limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return events[start_offset:]
    return events[start_offset : start_offset + limit]


def apply_replay_event(session: Session, event: dict[str, Any], batch: ReplayBatch | None = None) -> dict[str, Any]:
    existing = session.get(ProcessedEvent, event["event_id"])
    if existing is not None:
        return {"event_id": event["event_id"], "status": "duplicate"}

    payload = event.get("payload", {})
    order_state = event.get("order_state", {})
    order_payload = payload.get("order", {})

    if payload.get("customer"):
        _upsert(Customer, session, payload["customer"], "customer_id")

    for category in payload.get("category_translation", []):
        _upsert(CategoryTranslation, session, category, "product_category_name")

    for seller in payload.get("sellers", []):
        _upsert(Seller, session, seller, "seller_id")

    for product in payload.get("products", []):
        _upsert(Product, session, product, "product_id")

    order_model = session.get(Order, event["order_id"])
    if order_model is None:
        order_model = Order(order_id=event["order_id"])
        session.add(order_model)

    if order_payload:
        order_model.customer_id = order_payload.get("customer_id")
        order_model.order_estimated_delivery_date = _to_datetime(order_payload.get("order_estimated_delivery_date"))

    for field, value in order_state.items():
        if hasattr(order_model, field):
            setattr(order_model, field, _to_datetime(value) if "date" in field or "timestamp" in field or field.endswith("_at") else value)

    if event["event_type"] == "new_order":
        for item in payload.get("items", []):
            item_payload = dict(item)
            item_payload["shipping_limit_date"] = _to_datetime(item_payload.get("shipping_limit_date"))
            _upsert(OrderItem, session, item_payload, "order_item_key")

    if event["event_type"] == "payment":
        for payment in payload.get("payments", []):
            _upsert(Payment, session, payment, "payment_id")

    if event["event_type"] in {"delivered", "cancelled"}:
        for review in payload.get("reviews", []):
            review_payload = dict(review)
            review_payload["review_creation_date"] = _to_datetime(review_payload.get("review_creation_date"))
            review_payload["review_answer_timestamp"] = _to_datetime(review_payload.get("review_answer_timestamp"))
            _upsert(Review, session, review_payload, "review_key")

    processed_event = ProcessedEvent(
        event_id=event["event_id"],
        batch=batch,
        order_id=event["order_id"],
        event_type=event["event_type"],
        event_time=_to_datetime(event.get("event_time")),
        status="processed",
        payload_json=json.dumps(event.get("payload", {}), ensure_ascii=False),
    )
    session.add(processed_event)
    session.flush()
    return {"event_id": event["event_id"], "status": "processed"}


def run_direct_replay(
    session: Session,
    *,
    start_offset: int = 0,
    limit: int | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    events = _batch_window(load_event_bank(), start_offset, limit)

    batch = ReplayBatch(
        transport="direct",
        status="running",
        start_offset=start_offset,
        requested_limit=limit,
        pace_ms=pace_ms,
        total_events=len(events),
        started_at=datetime.utcnow(),
    )
    session.add(batch)
    session.commit()
    session.refresh(batch)

    processed = 0
    duplicates = 0
    errors = 0
    recent_results = []

    try:
        for event in events:
            try:
                result = apply_replay_event(session, event, batch=batch)
                session.commit()
                if result["status"] == "duplicate":
                    duplicates += 1
                else:
                    processed += 1
                recent_results.append(result)
            except Exception as exc:  # pragma: no cover - surfaced via API/tests
                session.rollback()
                errors += 1
                recent_results.append({"event_id": event["event_id"], "status": "failed", "error": str(exc)})
                failed_event = ProcessedEvent(
                    event_id=event["event_id"],
                    batch_id=batch.id,
                    order_id=event["order_id"],
                    event_type=event["event_type"],
                    event_time=_to_datetime(event.get("event_time")),
                    status="failed",
                    error_message=str(exc),
                    payload_json=json.dumps(event.get("payload", {}), ensure_ascii=False),
                )
                session.add(failed_event)
                session.commit()
            if pace_ms > 0:
                time.sleep(pace_ms / 1000)

        batch.status = "completed" if errors == 0 else "completed_with_errors"
        batch.processed_events = processed
        batch.failed_events = errors
        batch.completed_at = datetime.utcnow()
        session.commit()
    except Exception as exc:  # pragma: no cover - surfaced via API/tests
        session.rollback()
        batch.status = "failed"
        batch.last_error = str(exc)
        batch.completed_at = datetime.utcnow()
        session.add(batch)
        session.commit()
        raise

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "processed_events": processed,
        "duplicate_events": duplicates,
        "failed_events": errors,
        "total_events": len(events),
        "recent_results": recent_results[-10:],
    }


def reset_replay_state(session: Session) -> None:
    session.query(ProcessedEvent).delete()
    session.query(ReplayBatch).delete()
    session.query(Review).delete()
    session.query(Payment).delete()
    session.query(OrderItem).delete()
    session.query(Order).delete()
    session.query(Product).delete()
    session.query(Seller).delete()
    session.query(Customer).delete()
    session.query(CategoryTranslation).delete()
    session.query(Geolocation).delete()
    session.commit()


def get_batch_history(session: Session, limit: int = 10) -> list[dict[str, Any]]:
    batches = (
        session.execute(select(ReplayBatch).order_by(ReplayBatch.created_at.desc()).limit(limit))
        .scalars()
        .all()
    )
    return [
        {
            "id": batch.id,
            "transport": batch.transport,
            "status": batch.status,
            "topic": batch.topic,
            "created_at": batch.created_at.isoformat() if batch.created_at else None,
            "started_at": batch.started_at.isoformat() if batch.started_at else None,
            "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
            "processed_events": batch.processed_events,
            "failed_events": batch.failed_events,
            "total_events": batch.total_events,
            "requested_limit": batch.requested_limit,
        }
        for batch in batches
    ]


def get_recent_activity(session: Session, limit: int = 20) -> list[dict[str, Any]]:
    rows = (
        session.execute(select(ProcessedEvent).order_by(ProcessedEvent.processed_at.desc()).limit(limit))
        .scalars()
        .all()
    )
    return [
        {
            "event_id": row.event_id,
            "order_id": row.order_id,
            "event_type": row.event_type,
            "status": row.status,
            "event_time": row.event_time.isoformat() if row.event_time else None,
            "processed_at": row.processed_at.isoformat() if row.processed_at else None,
        }
        for row in rows
    ]


def _build_order_dataset(session: Session) -> list[dict[str, Any]]:
    orders = session.execute(select(Order)).scalars().all()
    if not orders:
        return []

    customers = {row.customer_id: row for row in session.execute(select(Customer)).scalars().all()}
    items_by_order: defaultdict[str, list[OrderItem]] = defaultdict(list)
    for item in session.execute(select(OrderItem)).scalars().all():
        items_by_order[item.order_id].append(item)

    payments_by_order: defaultdict[str, list[Payment]] = defaultdict(list)
    for payment in session.execute(select(Payment)).scalars().all():
        payments_by_order[payment.order_id].append(payment)

    products = {row.product_id: row for row in session.execute(select(Product)).scalars().all()}
    translations = {
        row.product_category_name: row.product_category_name_english
        for row in session.execute(select(CategoryTranslation)).scalars().all()
    }
    reviews_by_order: defaultdict[str, list[Review]] = defaultdict(list)
    for review in session.execute(select(Review)).scalars().all():
        reviews_by_order[review.order_id].append(review)

    rows = []
    for order in orders:
        order_items = items_by_order.get(order.order_id, [])
        payments = payments_by_order.get(order.order_id, [])
        revenue = sum((item.price or 0) + (item.freight_value or 0) for item in order_items)
        paid = sum(payment.payment_value or 0 for payment in payments)
        categories = []
        for item in order_items:
            product = products.get(item.product_id)
            if product and product.product_category_name:
                categories.append(
                    translations.get(product.product_category_name, product.product_category_name)
                )
        customer = customers.get(order.customer_id)
        rows.append(
            {
                "order_id": order.order_id,
                "status": order.order_status,
                "purchase_ts": order.order_purchase_timestamp,
                "delivered_ts": order.order_delivered_customer_date,
                "customer_unique_id": customer.customer_unique_id if customer else None,
                "customer_city": customer.customer_city if customer else None,
                "customer_state": customer.customer_state if customer else None,
                "revenue": revenue,
                "payment_value": paid,
                "item_count": len(order_items),
                "categories": categories,
                "review_score": (
                    sum(review.review_score for review in reviews_by_order[order.order_id] if review.review_score is not None)
                    / len([review for review in reviews_by_order[order.order_id] if review.review_score is not None])
                    if any(review.review_score is not None for review in reviews_by_order[order.order_id])
                    else None
                ),
            }
        )
    return rows


def get_dashboard_snapshot(session: Session) -> dict[str, Any]:
    order_rows = _build_order_dataset(session)
    if not order_rows:
        return {
            "kpis": {
                "orders": 0,
                "revenue": 0.0,
                "delivered_orders": 0,
                "cancelled_orders": 0,
                "avg_order_value": 0.0,
                "unique_customers": 0,
                "avg_delivery_days": None,
                "avg_satisfaction_score": None,
            },
            "trends": [],
            "top_categories": [],
            "recent_orders": [],
            "replay": get_batch_history(session, limit=5),
            "activity": get_recent_activity(session, limit=10),
        }

    delivered_orders = sum(1 for row in order_rows if row["status"] == "delivered")
    cancelled_orders = sum(1 for row in order_rows if row["status"] == "cancelled")
    revenue = round(sum(row["payment_value"] or row["revenue"] for row in order_rows), 2)
    unique_customers = len({row["customer_unique_id"] for row in order_rows if row["customer_unique_id"]})
    avg_order_value = round(revenue / len(order_rows), 2) if order_rows else 0.0

    delivery_durations = []
    for row in order_rows:
        purchase = row.get("purchase_ts")
        delivered = row.get("delivered_ts")
        if purchase and delivered:
            delta = (delivered - purchase).total_seconds() / 86400
            if 0 < delta < 120:
                delivery_durations.append(delta)
    avg_delivery_days = round(sum(delivery_durations) / len(delivery_durations), 1) if delivery_durations else None

    review_scores = [row["review_score"] for row in order_rows if row.get("review_score") is not None]
    avg_satisfaction = round(sum(review_scores) / len(review_scores), 2) if review_scores else None

    monthly = defaultdict(lambda: {"orders": 0, "revenue": 0.0, "delivered": 0, "cancelled": 0})
    category_totals: Counter[str] = Counter()
    for row in order_rows:
        if row["purchase_ts"]:
            period = row["purchase_ts"].strftime("%Y-%m")
            monthly[period]["orders"] += 1
            monthly[period]["revenue"] += row["payment_value"] or row["revenue"]
            if row["status"] == "delivered":
                monthly[period]["delivered"] += 1
            if row["status"] == "cancelled":
                monthly[period]["cancelled"] += 1
        for category in row["categories"]:
            category_totals[category] += row["revenue"] / max(len(row["categories"]), 1)

    trends = [
        {"period": period, **values, "revenue": round(values["revenue"], 2)}
        for period, values in sorted(monthly.items())
    ]

    top_categories = [
        {"category": category, "revenue": round(value, 2)}
        for category, value in category_totals.most_common(10)
    ]

    recent_orders = sorted(
        order_rows,
        key=lambda row: row["purchase_ts"] or datetime.min,
        reverse=True,
    )[:12]
    for row in recent_orders:
        row["purchase_ts"] = row["purchase_ts"].isoformat() if row["purchase_ts"] else None
        row["delivered_ts"] = row["delivered_ts"].isoformat() if row["delivered_ts"] else None
        row["revenue"] = round(row["revenue"], 2)
        row["payment_value"] = round(row["payment_value"], 2)

    return {
        "kpis": {
            "orders": len(order_rows),
            "revenue": revenue,
            "delivered_orders": delivered_orders,
            "cancelled_orders": cancelled_orders,
            "avg_order_value": avg_order_value,
            "unique_customers": unique_customers,
            "avg_delivery_days": avg_delivery_days,
            "avg_satisfaction_score": avg_satisfaction,
        },
        "trends": trends[-12:],
        "top_categories": top_categories,
        "recent_orders": recent_orders,
        "replay": get_batch_history(session, limit=5),
        "activity": get_recent_activity(session, limit=10),
    }


def get_system_health(session: Session) -> dict[str, Any]:
    metadata = ensure_event_bank()
    processed_total = session.scalar(select(func.count()).select_from(ProcessedEvent)) or 0
    order_total = session.scalar(select(func.count()).select_from(Order)) or 0
    payment_total = session.scalar(select(func.count()).select_from(Payment)) or 0
    batch_total = session.scalar(select(func.count()).select_from(ReplayBatch)) or 0
    return {
        "database_url": "sqlite" if "sqlite" in str(session.bind.url) else "mysql",
        "source_data_present": all((DATA_DIR / filename).exists() for filename in SOURCE_FILES.values()),
        "event_bank": metadata,
        "database_counts": {
            "orders": order_total,
            "payments": payment_total,
            "processed_events": processed_total,
            "replay_batches": batch_total,
        },
    }


def list_ml_models() -> list[dict[str, Any]]:
    models = []
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        model_name = path.stem
        try:
            package = pickle.loads(path.read_bytes())
            models.append(
                {
                    "model_name": model_name,
                    "target": package.get("target"),
                    "score": package.get("score"),
                    "feature_count": len(package.get("feature_names") or []),
                    "supports_prediction": model_name in PRIMARY_TARGET,
                }
            )
        except Exception:
            models.append(
                {
                    "model_name": model_name,
                    "error": "Failed to load model artifact.",
                    "supports_prediction": False,
                }
            )
    return models


def _load_model_package(model_name: str) -> dict[str, Any]:
    """Load a model package by looking up from filesystem-enumerated paths only."""
    known = {p.stem: p for p in MODEL_DIR.glob("*.pkl")}
    path = known.get(model_name)
    if path is None:
        raise FileNotFoundError(f"Unknown model '{model_name}'")
    return pickle.loads(path.read_bytes())


def predict_from_training_sample(model_name: str, row_index: int = 0) -> dict[str, Any]:
    if model_name not in PRIMARY_TARGET:
        raise ValueError(f"Model '{model_name}' does not expose supervised predictions.")

    known_datasets = {p.stem: p for p in ML_DATA_DIR.glob("*.csv")}
    dataset_path = known_datasets.get(model_name)
    if dataset_path is None:
        raise FileNotFoundError(f"Training dataset not found for model '{model_name}'")

    package = _load_model_package(model_name)
    dataset = pd.read_csv(dataset_path)
    if dataset.empty:
        raise ValueError(f"Training dataset for '{model_name}' is empty.")

    row_index = max(0, min(row_index, len(dataset) - 1))
    target_column = PRIMARY_TARGET[model_name]
    row = dataset.iloc[[row_index]].copy()
    target_value = row[target_column].iloc[0] if target_column in row.columns else None
    feature_input = row.drop(columns=[target_column], errors="ignore")

    artifact = preprocess_data(
        df=feature_input,
        fit=False,
        encoder=package.get("encoder"),
        scaler=package.get("scaler"),
        fill_values=package.get("fill_values"),
        clipping_values=package.get("clipping_values"),
        **PREPROCESS_CONFIG[model_name],
    )
    feature_frame = artifact["data"]
    expected_features = package.get("feature_names") or feature_frame.columns.tolist()
    feature_frame = feature_frame.reindex(columns=expected_features, fill_value=0)

    model = package["model"]
    prediction = model.predict(feature_frame)[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[0].tolist()

    return {
        "model_name": model_name,
        "row_index": row_index,
        "prediction": float(prediction) if isinstance(prediction, (int, float)) else str(prediction),
        "target": target_value,
        "probabilities": probabilities,
        "score": package.get("score"),
        "feature_count": len(expected_features),
    }


def get_feature_summary(session: Session) -> dict[str, Any]:
    """Summarise the incremental feature store: customer, seller, and product aggregates."""
    from database import Seller

    customers = session.execute(select(Customer)).scalars().all()
    sellers = session.execute(select(Seller)).scalars().all()
    products = session.execute(select(Product)).scalars().all()
    payments_all = session.execute(select(Payment)).scalars().all()
    items_all = session.execute(select(OrderItem)).scalars().all()
    orders_all = session.execute(select(Order)).scalars().all()

    payments_by_order: defaultdict[str, float] = defaultdict(float)
    for p in payments_all:
        payments_by_order[p.order_id] += p.payment_value or 0

    orders_by_customer: defaultdict[str, list] = defaultdict(list)
    for o in orders_all:
        if o.customer_id:
            orders_by_customer[o.customer_id].append(o)

    customer_lifetime: list[dict[str, Any]] = []
    for customer in customers[:20]:
        cid = customer.customer_id
        corders = orders_by_customer.get(cid, [])
        total_revenue = sum(payments_by_order.get(o.order_id, 0) for o in corders)
        customer_lifetime.append(
            {
                "customer_id": cid[:8] + "…",
                "state": customer.customer_state,
                "order_count": len(corders),
                "lifetime_revenue": round(total_revenue, 2),
            }
        )
    customer_lifetime.sort(key=lambda row: row["lifetime_revenue"], reverse=True)

    items_by_seller: defaultdict[str, list] = defaultdict(list)
    for item in items_all:
        if item.seller_id:
            items_by_seller[item.seller_id].append(item)

    seller_summary: list[dict[str, Any]] = []
    for seller in sellers[:20]:
        sid = seller.seller_id
        sitems = items_by_seller.get(sid, [])
        revenue = sum((i.price or 0) for i in sitems)
        seller_summary.append(
            {
                "seller_id": sid[:8] + "…",
                "state": seller.seller_state,
                "order_count": len({i.order_id for i in sitems}),
                "revenue": round(revenue, 2),
            }
        )
    seller_summary.sort(key=lambda row: row["revenue"], reverse=True)

    return {
        "customer_count": len(customers),
        "seller_count": len(sellers),
        "product_count": len(products),
        "top_customers": customer_lifetime[:10],
        "top_sellers": seller_summary[:10],
    }


def process_excel_upload(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Accept an uploaded Excel or CSV file, validate it, and return a preview.

    The original bytes are never written to the source data directory; they are
    read into memory only so the originals remain untouched.
    """
    import io

    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif lower.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError(f"Unsupported file type: {filename!r}. Upload a .csv, .xls, or .xlsx file.")
    except Exception as exc:
        raise ValueError(f"Could not parse uploaded file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no rows.")

    preview_rows = df.head(5).where(pd.notnull(df.head(5)), other=None).to_dict(orient="records")
    null_counts = df.isnull().sum()
    columns_with_nulls = [col for col in df.columns if null_counts[col] > 0]

    return {
        "filename": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "column_count": len(df.columns),
        "preview": preview_rows,
        "null_columns": columns_with_nulls,
        "validation": "passed" if not df.empty else "empty",
        "note": "Original file has not been modified. This is a preview of the uploaded data only.",
    }
