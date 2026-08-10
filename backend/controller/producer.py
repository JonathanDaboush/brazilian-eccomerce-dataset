from services.kafka_service import ReplayProducer


_producer = ReplayProducer()


def publish_next_batch(batch_size: int | None = None, replay_speed_ms: int | None = None):
    return _producer.publish_batch(batch_size=batch_size, replay_speed_ms=replay_speed_ms)
