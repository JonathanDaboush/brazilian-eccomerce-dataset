import os
import glob
import pandas as pd

from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy import inspect
from dotenv import load_dotenv


load_dotenv()


# ==========================
# DATABASE CONNECTION
# ==========================

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)


engine = sqlalchemy_create_engine(DATABASE_URL)


CSV_FOLDER = "original_data"


def clean_table_name(filename):

    name = os.path.basename(filename)

    name = name.replace(".csv", "")
    name = name.lower()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    return name



def database_has_data():

    inspector = inspect(engine)

    tables = inspector.get_table_names()

    # If no tables exist, database is empty
    if not tables:
        return False

    return True



def load_all_csv():

    # Prevent duplicate loading
    if database_has_data():

        print("Database already contains tables. Skipping CSV load.")
        return


    csv_files = glob.glob(
        os.path.join(CSV_FOLDER, "*.csv")
    )


    if not csv_files:
        print("No CSV files found.")
        return


    print(f"Found {len(csv_files)} CSV files")


    for file in csv_files:

        try:

            table_name = clean_table_name(file)

            print(f"\nLoading {file}")

            df = pd.read_csv(
                file,
                low_memory=False
            )

            print(f"Rows: {len(df)}")


            df.to_sql(
                name=table_name,
                con=engine,
                if_exists="replace",
                index=False,
                chunksize=5000
            )


            print(
                f"SUCCESS: {table_name} inserted"
            )


        except Exception as e:

            print(
                f"FAILED loading {file}"
            )

            print(e)