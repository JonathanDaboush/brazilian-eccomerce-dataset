from __future__ import annotations

import argparse
import json
import logging
import os

from database import SessionLocal, init_db
from services import apply_replay_event, ensure_event_bank


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC", "olist-orders")
DEFAULT_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
DEFAULT_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "olist-replay-consumer")


def build_consumer(
    *,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    topic: str = DEFAULT_TOPIC,
    group_id: str = DEFAULT_GROUP,
) -> KafkaConsumer:
    from kafka import KafkaConsumer

    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=group_id,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )


def consume_forever(
    *,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    topic: str = DEFAULT_TOPIC,
    group_id: str = DEFAULT_GROUP,
) -> None:
    init_db()
    ensure_event_bank()
    consumer = build_consumer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        group_id=group_id,
    )

    session = SessionLocal()
    try:
        for message in consumer:
            try:
                result = apply_replay_event(session, message.value)
                session.commit()
                consumer.commit()
                LOGGER.info(
                    "Processed Kafka event %s for order %s (%s)",
                    result["event_id"],
                    message.value.get("order_id"),
                    result["status"],
                )
            except Exception as exc:  # pragma: no cover - operational path
                session.rollback()
                LOGGER.exception("Failed to process Kafka event: %s", exc)
    finally:
        session.close()
        consumer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume Olist replay events from Kafka.")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--group-id", default=DEFAULT_GROUP)
    args = parser.parse_args()
    consume_forever(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
    )


if __name__ == "__main__":
    main()
