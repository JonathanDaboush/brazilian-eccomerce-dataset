import json
import logging
import os

from kafka import KafkaConsumer

from services.replay_service import EVENT_TOPIC, append_log, ensure_replay_schema, process_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("olist-consumer")


def build_consumer() -> KafkaConsumer:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    group_id = os.getenv("KAFKA_CONSUMER_GROUP", "olist-replay-consumer")
    return KafkaConsumer(
        EVENT_TOPIC,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=1000,
    )


def run_consumer_loop() -> None:
    ensure_replay_schema()
    consumer = build_consumer()
    logger.info("Kafka consumer started for topic=%s", EVENT_TOPIC)

    while True:
        try:
            batch = consumer.poll(timeout_ms=1000, max_records=100)
            if not batch:
                continue

            for _tp, messages in batch.items():
                for message in messages:
                    payload = message.value
                    required = {"event_id", "order_id", "event_type", "payload"}
                    if not isinstance(payload, dict) or not required.issubset(payload.keys()):
                        append_log("failed", f"Malformed message skipped: {payload}")
                        logger.error("Malformed message: %s", payload)
                        continue

                    process_event(payload)

            consumer.commit()
        except Exception as exc:
            logger.exception("Consumer processing failure")
            append_log("failed", str(exc))


if __name__ == "__main__":
    run_consumer_loop()
