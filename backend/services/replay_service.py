import json
import logging
import os
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database import engine

logger = logging.getLogger(__name__)

EVENT_TOPIC = os.getenv("KAFKA_TOPIC", "olist-orders-events")


def _exec(sql: str, params: dict[str, Any] | None = None):
    with engine.begin() as conn:
        return conn.execute(text(sql), params or {})


def ensure_replay_schema() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS event_bank_events (
            event_id VARCHAR(120) PRIMARY KEY,
            order_id VARCHAR(64) NOT NULL,
            event_type VARCHAR(32) NOT NULL,
            event_timestamp DATETIME NULL,
            event_key VARCHAR(64) NOT NULL,
            payload_json JSON NOT NULL,
            source_hash VARCHAR(120) NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_event_bank_order (order_id),
            INDEX idx_event_bank_time (event_timestamp)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS replay_state (
            state_id INT PRIMARY KEY,
            status VARCHAR(24) NOT NULL,
            current_offset BIGINT NOT NULL DEFAULT 0,
            batch_size INT NOT NULL DEFAULT 200,
            replay_speed_ms INT NOT NULL DEFAULT 0,
            last_batch_produced INT NOT NULL DEFAULT 0,
            last_error TEXT NULL,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS replay_batches (
            batch_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            produced_events INT NOT NULL,
            requested_batch_size INT NOT NULL,
            replay_speed_ms INT NOT NULL,
            producer_status VARCHAR(24) NOT NULL,
            started_at DATETIME NOT NULL,
            finished_at DATETIME NULL,
            error_message TEXT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS replay_orders (
            order_id VARCHAR(64) PRIMARY KEY,
            customer_id VARCHAR(64) NOT NULL,
            order_status VARCHAR(32) NOT NULL,
            order_purchase_timestamp DATETIME NULL,
            order_approved_at DATETIME NULL,
            order_delivered_carrier_date DATETIME NULL,
            order_delivered_customer_date DATETIME NULL,
            order_estimated_delivery_date DATETIME NULL,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_replay_orders_customer (customer_id),
            INDEX idx_replay_orders_status (order_status)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS replay_order_items (
            order_id VARCHAR(64) NOT NULL,
            order_item_id INT NOT NULL,
            product_id VARCHAR(64) NOT NULL,
            seller_id VARCHAR(64) NOT NULL,
            price DOUBLE NULL,
            freight_value DOUBLE NULL,
            PRIMARY KEY (order_id, order_item_id),
            INDEX idx_replay_items_seller (seller_id),
            INDEX idx_replay_items_product (product_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS replay_payments (
            payment_id BIGINT PRIMARY KEY,
            order_id VARCHAR(64) NOT NULL,
            payment_type VARCHAR(32) NULL,
            payment_installments INT NULL,
            payment_value DOUBLE NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_replay_payments_order (order_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS replay_reviews (
            review_key BIGINT PRIMARY KEY,
            order_id VARCHAR(64) NOT NULL,
            review_score INT NULL,
            review_creation_date DATETIME NULL,
            review_answer_timestamp DATETIME NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_replay_reviews_order (order_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS processed_events (
            event_id VARCHAR(120) PRIMARY KEY,
            order_id VARCHAR(64) NOT NULL,
            event_type VARCHAR(32) NOT NULL,
            processed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_processed_order (order_id),
            INDEX idx_processed_time (processed_at)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS consumer_events_log (
            log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            event_id VARCHAR(120) NULL,
            order_id VARCHAR(64) NULL,
            event_type VARCHAR(32) NULL,
            status VARCHAR(16) NOT NULL,
            details TEXT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_consumer_log_time (created_at)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS customer_features (
            customer_id VARCHAR(64) PRIMARY KEY,
            total_orders INT NOT NULL DEFAULT 0,
            lifetime_revenue DOUBLE NOT NULL DEFAULT 0,
            average_basket_size DOUBLE NOT NULL DEFAULT 0,
            days_since_last_purchase INT NULL,
            last_purchase_at DATETIME NULL,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS seller_features (
            seller_id VARCHAR(64) PRIMARY KEY,
            revenue DOUBLE NOT NULL DEFAULT 0,
            order_count INT NOT NULL DEFAULT 0,
            cancelled_orders INT NOT NULL DEFAULT 0,
            cancellation_rate DOUBLE NOT NULL DEFAULT 0,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS product_features (
            product_id VARCHAR(64) PRIMARY KEY,
            sales INT NOT NULL DEFAULT 0,
            average_review_score DOUBLE NULL,
            average_delivery_time_days DOUBLE NULL,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
    ]
    for stmt in statements:
        _exec(stmt)

    _exec(
        """
        INSERT INTO replay_state (state_id, status, current_offset, batch_size, replay_speed_ms)
        VALUES (1, 'idle', 0, 200, 0)
        ON DUPLICATE KEY UPDATE state_id = state_id
        """
    )


def build_event_bank_if_missing() -> dict[str, Any]:
    ensure_replay_schema()
    count = _exec("SELECT COUNT(*) as c FROM event_bank_events").first()._mapping["c"]
    if count > 0:
        return {"created": False, "events": int(count)}

    logger.info("Creating immutable event bank from source Olist tables")
    event_definitions = [
        ("order_created", "order_purchase_timestamp", None),
        ("order_payment", "order_approved_at", None),
        ("order_shipped", "order_delivered_carrier_date", None),
        ("order_delivered", "order_delivered_customer_date", None),
        ("order_cancelled", "COALESCE(order_approved_at, order_purchase_timestamp)", "order_status = 'canceled'"),
    ]

    total_inserted = 0
    for event_type, ts_expr, extra_filter in event_definitions:
        filter_sql = f"AND {extra_filter}" if extra_filter else ""
        result = _exec(
            f"""
            INSERT IGNORE INTO event_bank_events (event_id, order_id, event_type, event_timestamp, event_key, payload_json, source_hash)
            SELECT
                CONCAT(order_id, ':{event_type}:', {ts_expr}) AS event_id,
                order_id,
                '{event_type}' AS event_type,
                {ts_expr} AS event_timestamp,
                order_id AS event_key,
                JSON_OBJECT(
                    'order_id', order_id,
                    'event_type', '{event_type}',
                    'event_timestamp', CAST({ts_expr} AS CHAR),
                    'order_status', order_status
                ) AS payload_json,
                SHA2(CONCAT(order_id, ':{event_type}:', {ts_expr}), 256) AS source_hash
            FROM orders
            WHERE {ts_expr} IS NOT NULL {filter_sql}
            """
        )
        total_inserted += int(result.rowcount or 0)

    total = _exec("SELECT COUNT(*) as c FROM event_bank_events").first()._mapping["c"]
    return {"created": True, "events": int(total), "inserted_now": total_inserted}


def get_replay_state() -> dict[str, Any]:
    ensure_replay_schema()
    row = _exec("SELECT * FROM replay_state WHERE state_id = 1").mappings().first()
    processed = _exec("SELECT COUNT(*) as c FROM processed_events").first()._mapping["c"]
    failed = _exec("SELECT COUNT(*) as c FROM consumer_events_log WHERE status='failed'").first()._mapping["c"]
    total = _exec("SELECT COUNT(*) as c FROM event_bank_events").first()._mapping["c"]

    recent_window = _exec(
        """
        SELECT
            COUNT(*) AS processed_last_5m,
            MIN(processed_at) AS window_start,
            MAX(processed_at) AS window_end
        FROM processed_events
        WHERE processed_at >= DATE_SUB(NOW(), INTERVAL 5 MINUTE)
        """
    ).mappings().first()
    processing_rate = 0.0
    if recent_window and recent_window["processed_last_5m"]:
        start = recent_window["window_start"]
        end = recent_window["window_end"]
        if start and end and start != end:
            processing_rate = round(
                float(recent_window["processed_last_5m"]) / max((end - start).total_seconds(), 1.0),
                2,
            )

    latest_batch = _exec(
        """
        SELECT batch_id, produced_events, requested_batch_size, replay_speed_ms, producer_status, started_at, finished_at, error_message
        FROM replay_batches
        ORDER BY batch_id DESC
        LIMIT 1
        """
    ).mappings().first()

    return {
        "status": row["status"],
        "current_offset": int(row["current_offset"]),
        "batch_size": int(row["batch_size"]),
        "replay_speed_ms": int(row["replay_speed_ms"]),
        "last_batch_produced": int(row["last_batch_produced"]),
        "last_error": row["last_error"],
        "events_processed": int(processed),
        "events_failed": int(failed),
        "events_total": int(total),
        "events_remaining": max(int(total) - int(processed), 0),
        "processing_rate_eps": processing_rate,
        "latest_batch": (
            {
                "batch_id": int(latest_batch["batch_id"]),
                "produced_events": int(latest_batch["produced_events"] or 0),
                "requested_batch_size": int(latest_batch["requested_batch_size"] or 0),
                "replay_speed_ms": int(latest_batch["replay_speed_ms"] or 0),
                "producer_status": latest_batch["producer_status"],
                "started_at": latest_batch["started_at"].isoformat() if latest_batch["started_at"] else None,
                "finished_at": latest_batch["finished_at"].isoformat() if latest_batch["finished_at"] else None,
                "error_message": latest_batch["error_message"],
            }
            if latest_batch
            else None
        ),
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


def update_replay_state(**kwargs):
    if not kwargs:
        return
    cols = ", ".join([f"{k} = :{k}" for k in kwargs])
    kwargs["state_id"] = 1
    _exec(f"UPDATE replay_state SET {cols} WHERE state_id = :state_id", kwargs)


def fetch_event_batch(offset: int, batch_size: int) -> list[dict[str, Any]]:
    rows = _exec(
        """
        SELECT event_id, order_id, event_type, event_timestamp, event_key, payload_json
        FROM event_bank_events
        ORDER BY event_timestamp, event_id
        LIMIT :limit OFFSET :offset
        """,
        {"limit": batch_size, "offset": offset},
    ).mappings().all()

    events = []
    for row in rows:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        events.append(
            {
                "event_id": row["event_id"],
                "order_id": row["order_id"],
                "event_type": row["event_type"],
                "event_timestamp": row["event_timestamp"].isoformat() if row["event_timestamp"] else None,
                "event_key": row["event_key"],
                "payload": payload,
            }
        )
    return events


def append_log(status: str, details: str, event_id: str | None = None, order_id: str | None = None, event_type: str | None = None):
    _exec(
        """
        INSERT INTO consumer_events_log(event_id, order_id, event_type, status, details)
        VALUES (:event_id, :order_id, :event_type, :status, :details)
        """,
        {
            "event_id": event_id,
            "order_id": order_id,
            "event_type": event_type,
            "status": status,
            "details": details[:5000],
        },
    )


def _load_order_context(order_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    order = _exec(
        """
        SELECT order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at,
               order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
        FROM orders
        WHERE order_id = :order_id
        """,
        {"order_id": order_id},
    ).mappings().first()
    if not order:
        raise ValueError(f"Order not found for event replay: {order_id}")

    items = _exec(
        """
        SELECT order_id, order_item_id, product_id, seller_id, price, freight_value
        FROM order_items
        WHERE order_id = :order_id
        """,
        {"order_id": order_id},
    ).mappings().all()

    payments = _exec(
        """
        SELECT payment_id, order_id, payment_type, payment_installments, payment_value
        FROM order_payments
        WHERE order_id = :order_id
        """,
        {"order_id": order_id},
    ).mappings().all()

    reviews = _exec(
        """
        SELECT review_key, order_id, review_score, review_creation_date, review_answer_timestamp
        FROM order_reviews
        WHERE order_id = :order_id
        """,
        {"order_id": order_id},
    ).mappings().all()

    return dict(order), [dict(x) for x in items], [dict(x) for x in payments], [dict(x) for x in reviews]


def _recompute_customer(customer_id: str):
    stats = _exec(
        """
        SELECT
            o.customer_id,
            COUNT(DISTINCT o.order_id) AS total_orders,
            COALESCE(SUM(p.payment_value), 0) AS lifetime_revenue,
            COALESCE(AVG(order_revenue.order_total), 0) AS average_basket_size,
            MAX(o.order_purchase_timestamp) AS last_purchase_at
        FROM replay_orders o
        LEFT JOIN replay_payments p ON p.order_id = o.order_id
        LEFT JOIN (
            SELECT order_id, SUM(price + COALESCE(freight_value,0)) AS order_total
            FROM replay_order_items GROUP BY order_id
        ) order_revenue ON order_revenue.order_id = o.order_id
        WHERE o.customer_id = :customer_id
        GROUP BY o.customer_id
        """,
        {"customer_id": customer_id},
    ).mappings().first()

    if not stats:
        return

    days_since = None
    if stats["last_purchase_at"]:
        delta = datetime.utcnow() - stats["last_purchase_at"]
        days_since = max(int(delta.total_seconds() // 86400), 0)

    _exec(
        """
        INSERT INTO customer_features(customer_id, total_orders, lifetime_revenue, average_basket_size, days_since_last_purchase, last_purchase_at)
        VALUES (:customer_id, :total_orders, :lifetime_revenue, :average_basket_size, :days_since, :last_purchase_at)
        ON DUPLICATE KEY UPDATE
            total_orders = VALUES(total_orders),
            lifetime_revenue = VALUES(lifetime_revenue),
            average_basket_size = VALUES(average_basket_size),
            days_since_last_purchase = VALUES(days_since_last_purchase),
            last_purchase_at = VALUES(last_purchase_at)
        """,
        {
            "customer_id": customer_id,
            "total_orders": int(stats["total_orders"] or 0),
            "lifetime_revenue": float(stats["lifetime_revenue"] or 0),
            "average_basket_size": float(stats["average_basket_size"] or 0),
            "days_since": days_since,
            "last_purchase_at": stats["last_purchase_at"],
        },
    )


def _recompute_sellers(order_id: str):
    sellers = _exec(
        "SELECT DISTINCT seller_id FROM replay_order_items WHERE order_id = :order_id",
        {"order_id": order_id},
    ).mappings().all()
    for s in sellers:
        seller_id = s["seller_id"]
        stats = _exec(
            """
            SELECT
                i.seller_id,
                COUNT(DISTINCT i.order_id) AS order_count,
                COALESCE(SUM(i.price + COALESCE(i.freight_value,0)), 0) AS revenue,
                SUM(CASE WHEN o.order_status = 'canceled' THEN 1 ELSE 0 END) AS cancelled_orders
            FROM replay_order_items i
            LEFT JOIN replay_orders o ON o.order_id = i.order_id
            WHERE i.seller_id = :seller_id
            GROUP BY i.seller_id
            """,
            {"seller_id": seller_id},
        ).mappings().first()
        if not stats:
            continue
        order_count = int(stats["order_count"] or 0)
        cancelled = int(stats["cancelled_orders"] or 0)
        rate = (cancelled / order_count) if order_count else 0.0
        _exec(
            """
            INSERT INTO seller_features(seller_id, revenue, order_count, cancelled_orders, cancellation_rate)
            VALUES (:seller_id, :revenue, :order_count, :cancelled_orders, :cancellation_rate)
            ON DUPLICATE KEY UPDATE
                revenue = VALUES(revenue),
                order_count = VALUES(order_count),
                cancelled_orders = VALUES(cancelled_orders),
                cancellation_rate = VALUES(cancellation_rate)
            """,
            {
                "seller_id": seller_id,
                "revenue": float(stats["revenue"] or 0),
                "order_count": order_count,
                "cancelled_orders": cancelled,
                "cancellation_rate": rate,
            },
        )


def _recompute_products(order_id: str):
    products = _exec(
        "SELECT DISTINCT product_id FROM replay_order_items WHERE order_id = :order_id",
        {"order_id": order_id},
    ).mappings().all()
    for p in products:
        product_id = p["product_id"]
        stats = _exec(
            """
            SELECT
                i.product_id,
                COUNT(*) AS sales,
                AVG(r.review_score) AS average_review_score,
                AVG(TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp, o.order_delivered_customer_date) / 24.0) AS avg_delivery
            FROM replay_order_items i
            LEFT JOIN replay_orders o ON o.order_id = i.order_id
            LEFT JOIN replay_reviews r ON r.order_id = i.order_id
            WHERE i.product_id = :product_id
            GROUP BY i.product_id
            """,
            {"product_id": product_id},
        ).mappings().first()
        if not stats:
            continue
        _exec(
            """
            INSERT INTO product_features(product_id, sales, average_review_score, average_delivery_time_days)
            VALUES (:product_id, :sales, :review, :avg_delivery)
            ON DUPLICATE KEY UPDATE
                sales = VALUES(sales),
                average_review_score = VALUES(average_review_score),
                average_delivery_time_days = VALUES(average_delivery_time_days)
            """,
            {
                "product_id": product_id,
                "sales": int(stats["sales"] or 0),
                "review": float(stats["average_review_score"]) if stats["average_review_score"] is not None else None,
                "avg_delivery": float(stats["avg_delivery"]) if stats["avg_delivery"] is not None else None,
            },
        )


def _event_status(event_type: str) -> str:
    return {
        "order_created": "created",
        "order_payment": "paid",
        "order_shipped": "shipped",
        "order_delivered": "delivered",
        "order_cancelled": "canceled",
    }.get(event_type, "created")


def _event_order_payload(order: dict[str, Any], event_type: str) -> dict[str, Any]:
    status = _event_status(event_type)
    return {
        "order_id": order.get("order_id"),
        "customer_id": order.get("customer_id"),
        "order_status": status,
        "purchase": order.get("order_purchase_timestamp"),
        "approved": order.get("order_approved_at") if event_type in {"order_payment", "order_shipped", "order_delivered", "order_cancelled"} else None,
        "carrier": order.get("order_delivered_carrier_date") if event_type in {"order_shipped", "order_delivered"} else None,
        "delivered": order.get("order_delivered_customer_date") if event_type == "order_delivered" else None,
        "estimated": order.get("order_estimated_delivery_date"),
    }


def process_event(event: dict[str, Any]) -> dict[str, Any]:
    ensure_replay_schema()
    event_id = event["event_id"]
    order_id = event["order_id"]
    event_type = event["event_type"]

    existing = _exec("SELECT 1 FROM processed_events WHERE event_id = :event_id", {"event_id": event_id}).first()
    if existing:
        append_log("skipped", "Duplicate event skipped", event_id, order_id, event_type)
        return {"status": "duplicate", "event_id": event_id}

    order, items, payments, reviews = _load_order_context(order_id)

    try:
        order_payload = _event_order_payload(order, event_type)
        _exec(
            """
            INSERT INTO replay_orders(order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at,
                                      order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date)
            VALUES (:order_id, :customer_id, :order_status, :purchase, :approved, :carrier, :delivered, :estimated)
            ON DUPLICATE KEY UPDATE
                customer_id = VALUES(customer_id),
                order_status = VALUES(order_status),
                order_purchase_timestamp = VALUES(order_purchase_timestamp),
                order_approved_at = VALUES(order_approved_at),
                order_delivered_carrier_date = VALUES(order_delivered_carrier_date),
                order_delivered_customer_date = VALUES(order_delivered_customer_date),
                order_estimated_delivery_date = VALUES(order_estimated_delivery_date)
            """,
            order_payload,
        )

        if event_type == "order_created":
            for item in items:
                _exec(
                    """
                    INSERT INTO replay_order_items(order_id, order_item_id, product_id, seller_id, price, freight_value)
                    VALUES (:order_id, :order_item_id, :product_id, :seller_id, :price, :freight)
                    ON DUPLICATE KEY UPDATE
                        product_id = VALUES(product_id),
                        seller_id = VALUES(seller_id),
                        price = VALUES(price),
                        freight_value = VALUES(freight_value)
                    """,
                    {
                        "order_id": order_id,
                        "order_item_id": item.get("order_item_id"),
                        "product_id": item.get("product_id"),
                        "seller_id": item.get("seller_id"),
                        "price": item.get("price"),
                        "freight": item.get("freight_value"),
                    },
                )

        if event_type == "order_payment":
            for payment in payments:
                _exec(
                    """
                    INSERT INTO replay_payments(payment_id, order_id, payment_type, payment_installments, payment_value)
                    VALUES (:payment_id, :order_id, :payment_type, :installments, :payment_value)
                    ON DUPLICATE KEY UPDATE
                        payment_type = VALUES(payment_type),
                        payment_installments = VALUES(payment_installments),
                        payment_value = VALUES(payment_value)
                    """,
                    {
                        "payment_id": payment.get("payment_id"),
                        "order_id": order_id,
                        "payment_type": payment.get("payment_type"),
                        "installments": payment.get("payment_installments"),
                        "payment_value": payment.get("payment_value"),
                    },
                )

        if event_type == "order_payment":
            _exec(
                """
                UPDATE replay_orders
                SET order_status='paid', order_approved_at=:approved
                WHERE order_id=:order_id
                """,
                {"order_id": order_id, "approved": order.get("order_approved_at")},
            )

        if event_type == "order_shipped":
            _exec(
                """
                UPDATE replay_orders
                SET order_status='shipped', order_delivered_carrier_date=:carrier
                WHERE order_id=:order_id
                """,
                {"order_id": order_id, "carrier": order.get("order_delivered_carrier_date")},
            )

        if event_type == "order_delivered":
            _exec(
                """
                UPDATE replay_orders
                SET order_status='delivered', order_delivered_customer_date=:delivered
                WHERE order_id=:order_id
                """,
                {"order_id": order_id, "delivered": order.get("order_delivered_customer_date")},
            )
            for review in reviews:
                _exec(
                    """
                    INSERT INTO replay_reviews(review_key, order_id, review_score, review_creation_date, review_answer_timestamp)
                    VALUES (:review_key, :order_id, :review_score, :review_creation_date, :review_answer_timestamp)
                    ON DUPLICATE KEY UPDATE
                        review_score = VALUES(review_score),
                        review_creation_date = VALUES(review_creation_date),
                        review_answer_timestamp = VALUES(review_answer_timestamp)
                    """,
                    {
                        "review_key": review.get("review_key"),
                        "order_id": order_id,
                        "review_score": review.get("review_score"),
                        "review_creation_date": review.get("review_creation_date"),
                        "review_answer_timestamp": review.get("review_answer_timestamp"),
                    },
                )

        if event_type == "order_cancelled":
            _exec("UPDATE replay_orders SET order_status='canceled' WHERE order_id=:order_id", {"order_id": order_id})

        customer_id = order.get("customer_id")
        if customer_id:
            _recompute_customer(customer_id)
        _recompute_sellers(order_id)
        _recompute_products(order_id)

        _exec(
            "INSERT INTO processed_events(event_id, order_id, event_type) VALUES (:event_id, :order_id, :event_type)",
            {"event_id": event_id, "order_id": order_id, "event_type": event_type},
        )
        append_log("processed", "Event processed successfully", event_id, order_id, event_type)
        return {"status": "processed", "event_id": event_id}
    except SQLAlchemyError as exc:
        append_log("failed", str(exc), event_id, order_id, event_type)
        raise


def get_dashboard_summary() -> dict[str, Any]:
    ensure_replay_schema()
    metrics = _exec(
        """
        SELECT
            COALESCE(SUM(p.payment_value), 0) AS revenue,
            SUM(CASE WHEN o.order_status IN ('created', 'approved', 'invoiced', 'processing', 'shipped') THEN 1 ELSE 0 END) AS active_orders,
            SUM(CASE WHEN o.order_status = 'delivered' THEN 1 ELSE 0 END) AS delivered_orders,
            SUM(CASE WHEN o.order_status = 'canceled' THEN 1 ELSE 0 END) AS cancelled_orders,
            AVG(CASE WHEN o.order_delivered_customer_date IS NOT NULL
                THEN TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp, o.order_delivered_customer_date) / 24.0
                ELSE NULL END) AS avg_delivery_days,
            AVG(r.review_score) AS avg_review_score,
            COUNT(DISTINCT o.customer_id) AS active_customers
        FROM replay_orders o
        LEFT JOIN replay_payments p ON p.order_id = o.order_id
        LEFT JOIN replay_reviews r ON r.order_id = o.order_id
        """
    ).mappings().first()

    activity = _exec(
        """
        SELECT event_type, COUNT(*) AS c
        FROM processed_events
        WHERE processed_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        GROUP BY event_type
        """
    ).mappings().all()

    freshness = _exec("SELECT MAX(processed_at) AS ts FROM processed_events").mappings().first()
    logs = _exec(
        """
        SELECT created_at, status, event_type, details
        FROM consumer_events_log
        ORDER BY created_at DESC
        LIMIT 20
        """
    ).mappings().all()
    replay = get_replay_state()

    return {
        "kpis": {
            "revenue": float(metrics["revenue"] or 0),
            "active_orders": int(metrics["active_orders"] or 0),
            "delivered_orders": int(metrics["delivered_orders"] or 0),
            "cancelled_orders": int(metrics["cancelled_orders"] or 0),
            "avg_delivery_days": float(metrics["avg_delivery_days"] or 0),
            "avg_review_score": float(metrics["avg_review_score"] or 0),
            "active_customers": int(metrics["active_customers"] or 0),
        },
        "recent_activity": [{"event_type": x["event_type"], "count_24h": int(x["c"])} for x in activity],
        "recent_logs": [
            {
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "status": row["status"],
                "event_type": row["event_type"],
                "details": row["details"],
            }
            for row in logs
        ],
        "data_freshness": freshness["ts"].isoformat() if freshness and freshness["ts"] else None,
        "processing_status": {
            "replay_status": replay["status"],
            "events_processed": replay["events_processed"],
            "events_remaining": replay["events_remaining"],
            "events_failed": replay["events_failed"],
            "processing_rate_eps": replay["processing_rate_eps"],
            "latest_batch": replay["latest_batch"],
            "last_error": replay["last_error"],
        },
    }


def get_trends(date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
    where_clauses = []
    params: dict[str, Any] = {}
    if date_from:
        where_clauses.append("DATE(o.order_purchase_timestamp) >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where_clauses.append("DATE(o.order_purchase_timestamp) <= :date_to")
        params["date_to"] = date_to
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    rows = _exec(
        f"""
        SELECT DATE(o.order_purchase_timestamp) AS day,
               COALESCE(SUM(p.payment_value), 0) AS revenue,
               SUM(CASE WHEN o.order_status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
               SUM(CASE WHEN o.order_status = 'canceled' THEN 1 ELSE 0 END) AS cancelled
        FROM replay_orders o
        LEFT JOIN replay_payments p ON p.order_id = o.order_id
        {where_sql}
        GROUP BY DATE(o.order_purchase_timestamp)
        ORDER BY day DESC
        LIMIT 90
        """,
        params,
    ).mappings().all()

    data = []
    for row in reversed(rows):
        data.append(
            {
                "day": row["day"].isoformat() if row["day"] else None,
                "revenue": float(row["revenue"] or 0),
                "delivered": int(row["delivered"] or 0),
                "cancelled": int(row["cancelled"] or 0),
            }
        )
    return {"series": data}
