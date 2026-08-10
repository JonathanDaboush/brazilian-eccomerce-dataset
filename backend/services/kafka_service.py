import json
import logging
import os
import time
from typing import Any

from kafka import KafkaProducer

from services.replay_service import EVENT_TOPIC, fetch_event_batch, get_replay_state, update_replay_state, _exec

logger = logging.getLogger(__name__)


class ReplayProducer:
    def __init__(self):
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            key_serializer=lambda x: x.encode("utf-8"),
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            retries=5,
            acks="all",
        )

    def publish_batch(self, batch_size: int | None = None, replay_speed_ms: int | None = None) -> dict[str, Any]:
        state = get_replay_state()
        if state["status"] == "paused":
            return {"status": "paused", "produced": 0}

        batch_size = int(batch_size or state["batch_size"])
        replay_speed_ms = int(replay_speed_ms if replay_speed_ms is not None else state["replay_speed_ms"])
        offset = int(state["current_offset"])

        events = fetch_event_batch(offset=offset, batch_size=batch_size)
        if not events:
            update_replay_state(status="completed", last_batch_produced=0)
            return {"status": "completed", "produced": 0, "offset": offset}

        started = time.time()
        batch_id = _exec(
            """
            INSERT INTO replay_batches(produced_events, requested_batch_size, replay_speed_ms, producer_status, started_at)
            VALUES (0, :requested_batch_size, :replay_speed_ms, 'running', NOW())
            """,
            {"requested_batch_size": batch_size, "replay_speed_ms": replay_speed_ms},
        ).lastrowid

        sent = 0
        try:
            for event in events:
                self.producer.send(EVENT_TOPIC, key=event["event_key"], value=event)
                sent += 1
                if replay_speed_ms > 0:
                    time.sleep(replay_speed_ms / 1000.0)

            self.producer.flush()
            update_replay_state(
                status="running",
                current_offset=offset + sent,
                batch_size=batch_size,
                replay_speed_ms=replay_speed_ms,
                last_batch_produced=sent,
                last_error=None,
            )
            _exec(
                """
                UPDATE replay_batches
                SET produced_events=:produced, producer_status='success', finished_at=NOW()
                WHERE batch_id=:batch_id
                """,
                {"produced": sent, "batch_id": batch_id},
            )
            return {
                "status": "running",
                "produced": sent,
                "next_offset": offset + sent,
                "duration_seconds": round(time.time() - started, 2),
            }
        except Exception as exc:
            logger.exception("Failed publishing batch")
            update_replay_state(status="failed", last_error=str(exc))
            _exec(
                """
                UPDATE replay_batches
                SET produced_events=:produced, producer_status='failed', finished_at=NOW(), error_message=:error
                WHERE batch_id=:batch_id
                """,
                {"produced": sent, "batch_id": batch_id, "error": str(exc)},
            )
            raise
