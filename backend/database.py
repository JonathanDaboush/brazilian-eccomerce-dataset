import os

from dotenv import load_dotenv

from sqlalchemy import create_engine as sqlalchemy_create_engine


load_dotenv()


MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


engine = sqlalchemy_create_engine(
        f"mysql+pymysql://"+
        f"{MYSQL_USER}:{MYSQL_PASSWORD}"+
        f"@{MYSQL_HOST}:{MYSQL_PORT}/"+
        f"{MYSQL_DATABASE}"
    )



def test_connection():

    try:

        with engine.connect():

            print("MySQL connected")

    except Exception as e:

        print("MySQL connection failed")
        print(e)



def close_engine():

    engine.dispose()