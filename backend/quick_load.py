from pathlib import Path

import pandas as pd

from database import engine


def load_csv_files():
    csv_directory = Path(__file__).resolve().parent / "original_data"
    if not csv_directory.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_directory}")

    for csv_file in csv_directory.glob("*.csv"):
        df = pd.read_csv(csv_file)
        table_name = csv_file.stem.lower()
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Loaded {table_name}: {len(df)} rows")


if __name__ == "__main__":
    load_csv_files()
