
from sqlalchemy import text
from database import engine
from producer import send_user_event



def get_users():

    with engine.connect() as connection:

        result = connection.execute(
            text("""
                SELECT *
                FROM users
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