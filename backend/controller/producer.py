from __future__ import annotations

import argparse
import json
import logging
import os
import time

from kafka import KafkaProducer

from services import ensure_event_bank, load_event_bank


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC", "olist-orders")
DEFAULT_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")


def build_producer(bootstrap_servers: str = DEFAULT_BOOTSTRAP) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        acks="all",
        retries=5,
        linger_ms=25,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def publish_events(
    *,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    topic: str = DEFAULT_TOPIC,
    start_offset: int = 0,
    limit: int | None = None,
    pace_ms: int = 0,
) -> dict:
    ensure_event_bank()
    events = load_event_bank()
    selected = events[start_offset:] if limit is None else events[start_offset : start_offset + limit]
    producer = build_producer(bootstrap_servers)

    sent = 0
    try:
        for event in selected:
            future = producer.send(
                topic,
                key=event["order_id"].encode("utf-8"),
                value=event,
            )
            future.get(timeout=30)
            sent += 1
            if pace_ms > 0:
                time.sleep(pace_ms / 1000)
        producer.flush()
    finally:
        producer.close()

    LOGGER.info("Published %s events to topic %s", sent, topic)
    return {"topic": topic, "published_events": sent, "start_offset": start_offset, "limit": limit}


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Olist replay events to Kafka.")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pace-ms", type=int, default=0)
    args = parser.parse_args()
    result = publish_events(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        start_offset=args.start_offset,
        limit=args.limit,
        pace_ms=args.pace_ms,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
