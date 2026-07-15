from pathlib import Path

import pandas as pd

from database import engine
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind=engine)

def load_csv_files():
    try:
        csv_directory = (
            Path(__file__).parent /
            "../original_data"
        )


        for csv_file in csv_directory.glob("*.csv"):

            print(
                f"Loading {csv_file.name}"
            )


            df = pd.read_csv(csv_file)


            table_name = csv_file.stem.lower()


            df.to_sql(
                table_name,
                engine,
                if_exists="replace",
                index=False
            )


            print(
                f"Loaded {table_name}"
            )
    except Exception as e:
        print(f"Error loading CSV files: {e}")
