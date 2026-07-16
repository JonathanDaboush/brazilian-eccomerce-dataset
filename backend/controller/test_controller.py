from sqlalchemy import text
from database import engine
from controller.producer import send_user_event
from quick_load import database_has_data, load_all_csv


def get_users():

    if not database_has_data():
        load_all_csv()

    with engine.connect() as connection:

        result = connection.execute(
            text("""
                SELECT *
                FROM customers
                LIMIT 100
            """)
        )

        rows = result.fetchall()


    users = [
        dict(row._mapping)
        for row in rows
    ]


    # Send every user into Kafka
    for user in users:
        send_user_event(user)


    return users