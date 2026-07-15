from kafka import KafkaConsumer
import json



consumer = KafkaConsumer(

    "users-topic",

    bootstrap_servers="localhost:9092",

    auto_offset_reset="earliest",

    value_deserializer=lambda x:
        json.loads(x.decode())

)



for message in consumer:

    user = message.value

    print(
        "Processing user:",
        user
    )