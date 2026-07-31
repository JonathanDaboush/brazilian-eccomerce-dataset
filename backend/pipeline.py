"""
OLIST E-COMMERCE ML PIPELINE
============================

Same two structural changes made here as in the retail star-schema
pipeline (retail_pipeline.py):

CHANGE 1 - ROW USAGE MANIFEST (replaces the old "olist_learning_sample /
olist_rest_data" full-table CSV export). Instead of dumping complete
copies of every table twice (once filtered to the sampled learning
orders, once filtered to everything else), the pipeline now writes one
CSV per raw table (customers, sellers, orders, order_items, products,
payments, reviews, geolocation, category_translation) listing every
primary-key value in that table together with a `used_in_training`
flag. For order-linked tables the flag is derived by checking whether
each row is reachable from an order in the current learning sample
(the same logic rebuild_related_tables() already uses). geolocation and
category_translation are static reference tables that are never
order-sampled - every row in them is available to every run, so they
are reported as fully used. Filter any *_row_usage.csv to
used_in_training == True to get exactly the rows this run consumed,
e.g. for deleting/archiving from your own staging tables. See
build_row_usage_manifest() / export_row_usage_manifest().

CHANGE 2 - INDEPENDENT, IMPORTABLE FUNCTIONS. The script used to run as
one long sequence of top-level statements that fire immediately on
`import`, and ingest_new_data_and_retrain() relied on `global df_orders,
df_customers, ...` plus `globals()[...]` assignment to update state.
Everything is now organized into small, single-purpose functions with
explicit inputs/outputs, plus two orchestration functions:

    run_initial_pipeline()                       -> context dict
    ingest_new_data_and_retrain(new_tables, ctx)  -> new context dict

`context` is a plain dict holding every intermediate artifact (cleaned
tables, the time-based train/future split, the learning/rest order
samples, engineered feature tables, processed/split datasets, the model
registry, trained model packages, the row-usage manifest, etc). Your
app can hold onto that dict and pass it straight back into
ingest_new_data_and_retrain() whenever new raw Olist rows show up - no
reliance on module globals.

Nothing runs automatically on import. The `if __name__ == "__main__":`
block at the bottom reproduces the original "run everything once"
script behavior for standalone use (`python olist_pipeline.py`).

All the bug fixes from the previous pass are preserved (no phantom
create_timestamp kwarg, correct "product_recommendation" registry key,
local model folder, future_orders always supplied, correct
"last_purchase" column name, cancellation dataset now merges
customer_city/state, no duplicated dataset-building blocks, order-level
sampling keeps every table relationally consistent).
"""

import numpy as np
import pandas as pd
import os
import glob
import json
import shutil
import pickle
import time

import kagglehub

from datetime import datetime
from scipy.sparse import csr_matrix

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
    GroupShuffleSplit
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# Small on purpose - this pipeline is tuned to run on a laptop, not a
# cluster. Bump this up if you're running it somewhere beefier.
DEFAULT_N_JOBS = 2

# How many orders (out of the full ~100k+ Olist dataset) to actually use
# for feature engineering / model training. Everything else is set aside
# as "rest data" and never touched by any modeling step. Lower this
# further if your machine is still struggling.
LEARNING_SAMPLE_SIZE = 5000

DATASET_SLUG = "olistbr/brazilian-ecommerce"
MODEL_SAVE_DIR = "olist_saved_models"
ML_TRAINING_FOLDER = "olist_ml_training"

# CHANGE 1: replaces the old LEARNING_FOLDER / REST_FOLDER full-table
# CSV export. Controls whether run_initial_pipeline() /
# ingest_new_data_and_retrain() write out the per-table row-usage
# manifest described above.
EXPORT_ROW_USAGE_MANIFEST = True
ROW_USAGE_FOLDER = "olist_row_usage_manifest"

LOCAL_OUTPUT_DIRS = (ML_TRAINING_FOLDER, MODEL_SAVE_DIR, ROW_USAGE_FOLDER)

# Which raw table each engineered dataset is keyed to, and the column
# that is actually used as y. "product_recommendation" has no target -
# it's unsupervised - so it is intentionally absent here.
PRIMARY_TARGET = {
    "delivery_delay": "late_delivery",
    "order_cancellation": "cancelled",
    "review_prediction": "review_score",
    "demand_forecasting": "units_sold",
    "customer_purchase_prediction": "future_purchase",
}

# Per final-dataset preprocessing recipe. Every one of these gets fed
# straight into preprocess_data(**config) - no hand-written,
# copy-pasted preprocessing block per dataset.
#
# NOTE on demand_forecasting: "month_date" is deliberately left OUT of
# date_columns here, so preprocess_data leaves it untouched (raw
# datetime) instead of converting/dropping it. That lets
# split_ml_dataset (below) use it purely for chronological ordering,
# then drop it from the returned feature matrices automatically - the
# same pattern TIME_COLUMNS/"date" uses in the retail pipeline.
PREPROCESS_CONFIG = {
    "delivery_delay": dict(
        date_columns=["order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"],
        categorical_columns=["customer_city", "customer_state"],
        drop_columns=[
            "order_id", "customer_id", "order_status",
            "order_delivered_carrier_date", "order_delivered_customer_date"
        ],
        log_columns=["total_price", "total_freight", "avg_weight", "product_volume"],
        clip_columns=["total_price", "total_freight"],
    ),
    "order_cancellation": dict(
        date_columns=["order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"],
        categorical_columns=["customer_city", "customer_state"],
        drop_columns=[
            "order_id", "customer_id", "order_status",
            "order_delivered_carrier_date", "order_delivered_customer_date"
        ],
        log_columns=["order_value", "freight_value"],
        clip_columns=["order_value", "freight_value"],
    ),
    "review_prediction": dict(
        date_columns=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ],
        # order_id ends in "_id" so it's auto-dropped before encoding ever
        # runs anyway - it never did anything as a categorical column.
        categorical_columns=[],
        drop_columns=[
            "review_id", "review_key", "customer_id",
            "review_comment_title", "review_comment_message",
            "review_creation_date", "review_answer_timestamp"
        ],
        log_columns=["total_price", "total_freight"],
        clip_columns=["total_price", "total_freight"],
    ),
    "demand_forecasting": dict(
        date_columns=None,  # month_date stays raw - see note above
        categorical_columns=["product_category_name"],
        drop_columns=["month", "product_id"],
        log_columns=["average_price", "product_weight_g"],
        clip_columns=["average_price"],
    ),
    "customer_purchase_prediction": dict(
        date_columns=None,
        categorical_columns=["customer_city", "customer_state"],
        # total_orders / total_spent / average_order_value are dropped
        # because they're redundant with (and would swamp) the engineered
        # average_item_price / purchase_frequency features; last_purchase
        # is a raw timestamp whose signal already lives in
        # days_since_last_purchase.
        drop_columns=[
            "customer_id", "customer_unique_id",
            "total_orders", "total_spent", "average_order_value",
            "last_purchase",
        ],
        log_columns=[],
        clip_columns=[],
    ),
}

# Per final-dataset column that must survive preprocessing untouched
# because it's needed for a time-ordered split. Dropped from X
# automatically once the split is done (mirrors TIME_COLUMNS in the
# retail pipeline).
TIME_COLUMNS = {
    "demand_forecasting": "month_date",
}

SPLIT_CONFIG = {
    "delivery_delay": dict(split_type="random"),
    "order_cancellation": dict(split_type="random"),
    "review_prediction": dict(split_type="random"),
    "demand_forecasting": dict(split_type="time", time_column=TIME_COLUMNS["demand_forecasting"]),
    "customer_purchase_prediction": dict(split_type="random"),
}

# Maps short table names (used everywhere downstream) to the raw
# Kaggle CSV file names.
REQUIRED_TABLES = {
    "customers": "olist_customers_dataset",
    "sellers": "olist_sellers_dataset",
    "reviews": "olist_order_reviews_dataset",
    "items": "olist_order_items_dataset",
    "products": "olist_products_dataset",
    "geo": "olist_geolocation_dataset",
    "category": "product_category_name_translation",
    "orders": "olist_orders_dataset",
    "payments": "olist_order_payments_dataset"
}

PRIMARY_KEYS = {
    "customers": "customer_id",
    "sellers": "seller_id",
    "products": "product_id",
    "orders": "order_id",
    "order_items": "order_item_key",
    "payments": "payment_id",
    "reviews": "review_key",
    "geolocation": "geolocation_zip_code_prefix",
    "category_translation": "product_category_name",
}

# Static reference tables that are never order-sampled - every run has
# every row available, so the row-usage manifest reports them as fully
# used instead of trying to partition them by order membership.
REFERENCE_ONLY_TABLES = {"geolocation", "category_translation"}

FOREIGN_KEYS = {
    "orders": {"customer_id": ("customers", "customer_id")},
    "order_items": {
        "order_id": ("orders", "order_id"),
        "product_id": ("products", "product_id"),
        "seller_id": ("sellers", "seller_id"),
    },
    "reviews": {"order_id": ("orders", "order_id")},
    "payments": {"order_id": ("orders", "order_id")},
}


# ============================================================
# 0. CLEAN UP OUTPUT FROM THE PREVIOUS RUN
# ============================================================

def cleanup_previous_run(dataset_slug=DATASET_SLUG, local_output_dirs=LOCAL_OUTPUT_DIRS):

    print("=" * 80)
    print("CLEANING UP OUTPUT FROM PREVIOUS RUN")
    print("=" * 80)

    for folder in local_output_dirs:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
            print(f"Removed local output folder: {folder}")
        else:
            print(f"No local output folder to remove: {folder}")

    kagglehub_cache_root = os.path.expanduser("~/.cache/kagglehub/datasets")
    dataset_cache_path = os.path.join(kagglehub_cache_root, *dataset_slug.split("/"))

    if os.path.exists(dataset_cache_path):
        shutil.rmtree(dataset_cache_path, ignore_errors=True)
        print(f"Removed cached kaggle dataset: {dataset_cache_path}")
    else:
        print(f"No cached kaggle dataset found at: {dataset_cache_path}")

    print("Cleanup complete.\n")


# ============================================================
# 1. EXTRACT: DOWNLOAD AND LOAD DATASETS
# ============================================================

def download_and_load_datasets(dataset_slug=DATASET_SLUG):
    """
    Downloads the Kaggle dataset and loads every CSV found in it into a
    dict keyed by file name (without extension).
    """

    path = kagglehub.dataset_download(dataset_slug)
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

    datasets = {}
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        datasets[file_name] = pd.read_csv(file)

    print("LOADED DATASETS\n")
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")

    return datasets


# ============================================================
# 2. CREATE DATAFRAME REFERENCES
# ============================================================

def extract_required_frames(datasets, required_tables=REQUIRED_TABLES):
    """
    Validates that every required table is present and returns a dict of
    independent copies keyed by the SHORT names used throughout the rest
    of the pipeline (customers, sellers, reviews, items, products, geo,
    category, orders, payments) rather than the raw Kaggle file names.
    """

    for name, table in required_tables.items():
        if table not in datasets:
            raise ValueError(f"Missing dataset: {table}")

    return {name: datasets[table].copy() for name, table in required_tables.items()}


# ============================================================
# 3 + 4. DATATYPE + NULL VALUE CLEANING, PRIMARY KEY CLEANING,
#        SURROGATE KEYS, GEO NORMALIZATION
# ============================================================

def clean_dataframe(df, text_fill_columns=None, numeric_median_columns=None):
    """
    General purpose cleaning, mirroring the retail pipeline's
    clean_dataframe: drop duplicates, kill inf values, auto-parse any
    "date"/"timestamp" column to datetime. Table-specific fills
    (blank review text, median product dimensions) are layered on top
    via the optional arguments instead of being scattered inline.
    """

    df = df.copy()

    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in df.columns:
        lc = col.lower()
        if "date" in lc or "timestamp" in lc:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in (text_fill_columns or []):
        if col in df.columns:
            df[col] = df[col].fillna("")

    for col in (numeric_median_columns or []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


def clean_all_raw_tables(frames):
    """
    Runs every table-specific cleaning step (clean_dataframe with the
    right fill columns, PK deduplication, surrogate key generation for
    tables with no natural single-column PK, and zip-code-level geo
    aggregation) and returns a single dict keyed by the same short
    table names used by PRIMARY_KEYS / FOREIGN_KEYS / rebuild_related_
    tables (customers, sellers, orders, order_items, products, payments,
    reviews, geolocation, category_translation).
    """

    df_orders = clean_dataframe(frames["orders"])
    df_order_reviews = clean_dataframe(
        frames["reviews"],
        text_fill_columns=["review_comment_title", "review_comment_message"]
    )

    df_products = frames["products"].copy()
    df_products["product_category_name"] = df_products["product_category_name"].fillna("unknown")
    df_products = clean_dataframe(
        df_products,
        numeric_median_columns=[
            "product_name_lenght", "product_description_lenght", "product_photos_qty",
            "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
        ]
    )

    df_customers = clean_dataframe(frames["customers"])
    df_sellers = clean_dataframe(frames["sellers"])
    df_order_items = clean_dataframe(frames["items"])
    df_geolocation = clean_dataframe(frames["geo"])
    df_category_trans = clean_dataframe(frames["category"])
    df_order_payments = clean_dataframe(frames["payments"])

    print("\nRAW TABLES CLEANED")
    for name, df in {
        "customers": df_customers, "sellers": df_sellers, "orders": df_orders,
        "items": df_order_items, "products": df_products, "payments": df_order_payments,
        "reviews": df_order_reviews, "geo": df_geolocation, "category": df_category_trans,
    }.items():
        print(f"{name}: {df.shape}")

    df_products = df_products.drop_duplicates(subset=["product_id"]).copy()
    df_customers = df_customers.drop_duplicates(subset=["customer_id"]).copy()
    df_sellers = df_sellers.drop_duplicates(subset=["seller_id"]).copy()
    df_category_trans = df_category_trans.drop_duplicates(subset=["product_category_name"]).copy()

    df_order_items = df_order_items.copy()
    df_order_items["order_item_key"] = np.arange(1, len(df_order_items) + 1)

    df_order_payments = df_order_payments.copy()
    df_order_payments["payment_id"] = np.arange(1, len(df_order_payments) + 1)

    df_geolocation = (
        df_geolocation
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg({
            "geolocation_lat": "mean",
            "geolocation_lng": "mean",
            "geolocation_city": "first",
            "geolocation_state": "first"
        })
    )

    df_order_reviews = df_order_reviews.drop_duplicates(subset=["review_id"], keep="last").copy()
    df_order_reviews["review_key"] = np.arange(1, len(df_order_reviews) + 1)

    return {
        "customers": df_customers,
        "sellers": df_sellers,
        "orders": df_orders,
        "order_items": df_order_items,
        "products": df_products,
        "payments": df_order_payments,
        "reviews": df_order_reviews,
        "geolocation": df_geolocation,
        "category_translation": df_category_trans,
    }


# ============================================================
# STAR-SCHEMA-STYLE VALIDATION
# Same shape as the retail pipeline: PRIMARY_KEYS / FOREIGN_KEYS
# metadata drive one reusable set of validators.
# ============================================================

def validate_primary_keys(tables):

    print("\nPRIMARY KEY CHECK")
    for table, key in PRIMARY_KEYS.items():
        if table not in tables or key not in tables[table].columns:
            continue
        df = tables[table]
        print(f"  {table}.{key}: duplicates={df[key].duplicated().sum()} null={df[key].isna().sum()}")


def validate_foreign_keys(tables):

    print("\nFOREIGN KEY CHECK")
    for child_table, relationships in FOREIGN_KEYS.items():
        if child_table not in tables:
            continue
        child_df = tables[child_table]
        for fk_col, (parent_table, parent_col) in relationships.items():
            if parent_table not in tables or fk_col not in child_df.columns:
                continue
            parent_df = tables[parent_table]
            missing = (~child_df[fk_col].isin(parent_df[parent_col])).sum()
            print(f"  {child_table}.{fk_col} -> {parent_table}.{parent_col} missing={missing}")


def validate_business_rules(tables):

    print("\nBUSINESS RULE CHECK")
    if "order_items" in tables:
        print("  Negative prices:", (tables["order_items"].price < 0).sum())
    if "payments" in tables:
        print("  Negative payments:", (tables["payments"].payment_value < 0).sum())
    if "orders" in tables:
        orders = tables["orders"]
        invalid = (orders.order_delivered_customer_date < orders.order_purchase_timestamp).sum()
        print("  Invalid deliveries:", invalid)


def validate_star_schema(tables, label="DATA"):

    print("\n" + "=" * 60)
    print(f"STAR SCHEMA VALIDATION: {label}")
    print("=" * 60)
    for name, df in tables.items():
        print(f"  {name}: {df.shape} nulls={df.isna().sum().sum()}")

    validate_primary_keys(tables)
    validate_foreign_keys(tables)
    validate_business_rules(tables)


# ============================================================
# 5. GLOBAL TIME BOUNDARY (LEAKAGE-SAFE SPLIT)
# ============================================================

def split_orders_by_time(df_orders, train_fraction=0.70):
    """
    Date-orders every order and splits it into a training pool (the
    earliest train_fraction share) and a held-out future pool (the
    remainder) - the boundary used both for the customer_purchase_
    prediction label and for keeping model evaluation leakage-safe.
    """

    ordered = df_orders.sort_values("order_purchase_timestamp").reset_index(drop=True)

    split_index = int(len(ordered) * train_fraction)
    train_end_date = ordered.iloc[split_index - 1]["order_purchase_timestamp"]

    orders_train_full = ordered.iloc[:split_index].copy()
    orders_future = ordered.iloc[split_index:].copy()

    return orders_train_full, orders_future, train_end_date


def rebuild_related_tables(order_subset, tables):
    """
    Given a subset of orders and the full cleaned table dict, pulls
    every related row (items, products, sellers, payments, reviews) so
    the resulting bundle is fully self-consistent (no dangling foreign
    keys). geolocation and category_translation are reference tables
    and are passed through in full, unfiltered.
    """

    order_ids = order_subset.order_id.unique()
    customer_ids = order_subset.customer_id.unique()

    item_subset = tables["order_items"][tables["order_items"].order_id.isin(order_ids)].copy()
    product_ids = item_subset.product_id.unique()
    seller_ids = item_subset.seller_id.unique()

    payment_subset = tables["payments"][tables["payments"].order_id.isin(order_ids)].copy()
    review_subset = tables["reviews"][tables["reviews"].order_id.isin(order_ids)].copy()

    return {
        "orders": order_subset,
        "customers": tables["customers"][tables["customers"].customer_id.isin(customer_ids)].copy(),
        "order_items": item_subset,
        "products": tables["products"][tables["products"].product_id.isin(product_ids)].copy(),
        "sellers": tables["sellers"][tables["sellers"].seller_id.isin(seller_ids)].copy(),
        "payments": payment_subset,
        "reviews": review_subset,
        "geolocation": tables["geolocation"],
        "category_translation": tables["category_translation"],
    }


# ============================================================
# 6. LAPTOP-SIZED SAMPLING
#    Splits orders_train_full into:
#      - "learning sample" -> everything modeling touches
#      - "rest data"       -> untouched leftover
# ============================================================

def sample_learning_orders(orders_train_full, orders_future,
                            sample_size=LEARNING_SAMPLE_SIZE, random_state=RANDOM_STATE):
    """
    Samples `sample_size` orders out of the training pool for this run's
    learning sample. "Rest" = whatever training orders weren't sampled
    plus the entire held-out future partition - nothing is discarded,
    it's just not used for training this run.
    """

    n_learn = min(sample_size, len(orders_train_full))
    learning_orders = orders_train_full.sample(n=n_learn, random_state=random_state).copy()
    leftover_train_orders = orders_train_full.drop(index=learning_orders.index).copy()

    rest_orders = pd.concat([leftover_train_orders, orders_future], ignore_index=True)

    return learning_orders, rest_orders


# ============================================================
# 7. ROW USAGE MANIFEST (replaces the old learning-sample / rest-data
#    full-table CSV export)
# ============================================================

def build_row_usage_manifest(learning_source, tables):
    """
    Builds a per-table manifest of exactly which primary-key values
    were (and were not) pulled into this run's learning sample.

    For order-linked tables (orders, customers, order_items, products,
    sellers, payments, reviews): used = PK values present in
    learning_source[table] (as returned by rebuild_related_tables);
    unused = every other PK value present in the full cleaned table.

    For reference-only tables (geolocation, category_translation):
    every row is available to every run regardless of order sampling,
    so they are reported as fully used with nothing marked unused.

    Returns: dict table_name -> {
        "pk_column": str, "used_ids": sorted list, "unused_ids": sorted list,
        "used_count": int, "unused_count": int,
    }
    """

    manifest = {}

    for table_name, pk_col in PRIMARY_KEYS.items():

        if table_name not in tables or pk_col not in tables[table_name].columns:
            continue

        full_ids = set(tables[table_name][pk_col].dropna())

        if table_name in REFERENCE_ONLY_TABLES:
            used_ids = full_ids
        else:
            learning_table = learning_source.get(table_name)
            used_ids = set(learning_table[pk_col].dropna()) if learning_table is not None else set()

        unused_ids = full_ids - used_ids

        manifest[table_name] = {
            "pk_column": pk_col,
            "used_ids": sorted(used_ids, key=str),
            "unused_ids": sorted(unused_ids, key=str),
            "used_count": len(used_ids),
            "unused_count": len(unused_ids),
        }

    return manifest


def export_row_usage_manifest(manifest, output_folder=ROW_USAGE_FOLDER):
    """
    Writes one CSV per table (<table_name>_row_usage.csv) with columns
    [pk_column, used_in_training], plus a row_usage_summary.json with
    just the counts. Filter any of the CSVs to used_in_training == True
    to get the exact rows this run already consumed - handy for safely
    deleting/archiving them from your own staging tables.
    """

    os.makedirs(output_folder, exist_ok=True)
    written_paths = []

    print("\n" + "=" * 80)
    print("ROW USAGE MANIFEST (used vs. not-yet-used rows, per table)")
    print("=" * 80)

    for table_name, info in manifest.items():

        pk_col = info["pk_column"]

        rows = [{pk_col: v, "used_in_training": True} for v in info["used_ids"]]
        rows += [{pk_col: v, "used_in_training": False} for v in info["unused_ids"]]

        manifest_df = pd.DataFrame(rows)
        path = os.path.join(output_folder, f"{table_name}_row_usage.csv")
        manifest_df.to_csv(path, index=False)
        written_paths.append(path)

        print(f"{table_name:20s} used={info['used_count']:>8}  "
              f"unused={info['unused_count']:>8}  -> {path}")

    summary = {
        table_name: {
            "pk_column": info["pk_column"],
            "used_count": info["used_count"],
            "unused_count": info["unused_count"],
        }
        for table_name, info in manifest.items()
    }

    summary_path = os.path.join(output_folder, "row_usage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    written_paths.append(summary_path)

    print(f"\nSummary written to: {summary_path}")
    print("Tip: filter each *_row_usage.csv to used_in_training == True to get")
    print("the exact rows already consumed by this run.")

    return written_paths


# ============================================================
# 8. FEATURE ENGINEERING FUNCTIONS
# (per-dataset business logic - this is the one place each engineered
# table's rules should live; everything downstream is generic)
# ============================================================

def add_date_features(df, column):
    df = df.copy(deep=False)
    df[column] = pd.to_datetime(df[column], errors="coerce")
    df[f"{column}_year"] = df[column].dt.year.astype("float32")
    df[f"{column}_month"] = df[column].dt.month.astype("float32")
    df[f"{column}_weekday"] = df[column].dt.weekday.astype("float32")
    df[f"{column}_quarter"] = df[column].dt.quarter.astype("float32")
    return df


def create_delivery_delay_dataset(df_orders, df_order_items, df_customers, df_products):
    df = df_orders.copy()

    delivered = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    estimated = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    df["late_delivery"] = (delivered > estimated).astype("int8")

    df = df.merge(
        df_customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id", how="left"
    )

    order_features = (
        df_order_items.groupby("order_id").agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            unique_products=("product_id", "nunique"),
            unique_sellers=("seller_id", "nunique")
        ).reset_index()
    )
    df = df.merge(order_features, on="order_id", how="left")

    product_features = (
        df_order_items
        .merge(
            df_products[["product_id", "product_weight_g", "product_length_cm",
                          "product_height_cm", "product_width_cm"]],
            on="product_id", how="left"
        )
        .groupby("order_id").agg(
            avg_weight=("product_weight_g", "mean"),
            avg_length=("product_length_cm", "mean"),
            avg_height=("product_height_cm", "mean"),
            avg_width=("product_width_cm", "mean")
        ).reset_index()
    )
    product_features["product_volume"] = (
        product_features["avg_length"] * product_features["avg_height"] * product_features["avg_width"]
    )
    df = df.merge(product_features, on="order_id", how="left")

    # Drop columns only known AFTER delivery / target-derived columns to prevent leakage
    df = df.drop(
        columns=[
            "order_delivered_customer_date",
            "order_delivered_carrier_date",
            "order_status"
        ],
        errors="ignore"
    )

    df = df.dropna(subset=["late_delivery"])
    return df


def create_cancellation_dataset(df_orders, df_order_items, df_customers):
    df = df_orders.copy()

    df["cancelled"] = (df["order_status"] == "canceled").astype("int8")

    df = add_date_features(df, "order_purchase_timestamp")

    df["approval_delay_hours"] = (
        pd.to_datetime(df["order_approved_at"], errors="coerce")
        - pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    ).dt.total_seconds() / 3600

    df = df.merge(
        df_customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id", how="left"
    )

    order_features = (
        df_order_items.groupby("order_id").agg(
            order_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            item_count=("order_item_id", "count")
        ).reset_index()
    )
    df = df.merge(order_features, on="order_id", how="left")

    customer_history = (
        df_orders.groupby("customer_id").size().reset_index(name="previous_orders")
    )
    df = df.merge(customer_history, on="customer_id", how="left")

    df = df.drop(
        columns=[
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "order_status"
        ],
        errors="ignore"
    )

    return df


def create_review_dataset(df_order_reviews, df_orders, df_order_items):
    df = df_order_reviews.copy()

    df = df.merge(
        df_orders[["order_id", "order_delivered_customer_date",
                   "order_purchase_timestamp", "order_estimated_delivery_date"]],
        on="order_id", how="left"
    )

    delivered = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    purchased = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    estimated = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")

    df["delivery_days"] = (delivered - purchased).dt.days
    df["delay_days"] = (delivered - estimated).dt.days

    order_features = (
        df_order_items.groupby("order_id").agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum")
        ).reset_index()
    )
    df = df.merge(order_features, on="order_id", how="left")

    df = df.drop(
        columns=["review_comment_title", "review_comment_message",
                 "review_creation_date", "review_answer_timestamp"],
        errors="ignore"
    )

    return df


def create_demand_dataset(df_order_items, df_orders, df_products):
    df = df_order_items.copy()

    df = df.merge(
        df_orders[["order_id", "order_purchase_timestamp"]],
        on="order_id", how="left"
    )

    df["month"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M").dt.to_timestamp()

    df = (
        df.groupby(["product_id", "month"]).agg(
            units_sold=("order_item_id", "count"),
            average_price=("price", "mean")
        ).reset_index()
    )

    df = df.sort_values(["product_id", "month"])

    df["previous_sales"] = df.groupby("product_id")["units_sold"].shift(1)
    df["rolling_sales_3_month"] = (
        df.groupby("product_id")["units_sold"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.merge(
        df_products[["product_id", "product_category_name", "product_weight_g"]],
        on="product_id", how="left"
    )

    df["month_date"] = pd.to_datetime(df["month"], errors="coerce")

    return df


def create_customer_purchase_dataset(df_customers, df_orders, df_order_items, future_orders):

    customers = df_customers.copy()
    orders = df_orders.copy()
    items = df_order_items.copy()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    customer_history = (
        orders.groupby("customer_id").agg(
            total_orders=("order_id", "count"),
            last_purchase=("order_purchase_timestamp", "max")
        ).reset_index()
    )
    customers = customers.merge(customer_history, on="customer_id", how="left")

    spending = (
        items
        .merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .groupby("customer_id").agg(
            total_spent=("price", "sum"),
            average_item_price=("price", "mean"),
            total_items=("order_item_id", "count")
        ).reset_index()
    )
    customers = customers.merge(spending, on="customer_id", how="left")

    reference_date = orders["order_purchase_timestamp"].max()
    customers["days_since_last_purchase"] = (reference_date - customers["last_purchase"]).dt.days

    fill_zero = ["total_orders", "total_spent", "average_item_price", "total_items"]
    for col in fill_zero:
        customers[col] = customers[col].fillna(0)
    customers["days_since_last_purchase"] = customers["days_since_last_purchase"].fillna(9999)

    customers["average_order_value"] = (
        customers["total_spent"] / customers["total_orders"].replace(0, 1)
    )
    customers["purchase_frequency"] = (
        customers["total_orders"] / (customers["days_since_last_purchase"] + 1)
    )

    future_orders = future_orders.copy()
    future_customer_ids = (
        future_orders
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
        ["customer_unique_id"]
        .dropna()
        .unique()
    )
    customers["future_purchase"] = (
        customers["customer_unique_id"].isin(future_customer_ids).astype("int8")
    )

    return customers


def create_recommendation_dataset(df_order_items, df_orders):
    return (
        df_order_items
        .merge(df_orders[["order_id", "customer_id"]], on="order_id", how="left")
        .groupby(["customer_id", "product_id"])
        .agg(purchase_count=("order_id", "count"))
        .reset_index()
        .dropna()
    )


def create_all_ml_feature_tables(source, future_orders):
    """
    Builds every engineered dataset from one relationally-consistent
    `source` bundle (as returned by rebuild_related_tables), using
    dataset names that match PRIMARY_TARGET / PREPROCESS_CONFIG /
    MODEL_REGISTRY exactly.
    """

    return {
        "delivery_delay": create_delivery_delay_dataset(
            source["orders"], source["order_items"], source["customers"], source["products"]
        ),
        "order_cancellation": create_cancellation_dataset(
            source["orders"], source["order_items"], source["customers"]
        ),
        "review_prediction": create_review_dataset(
            source["reviews"], source["orders"], source["order_items"]
        ),
        "demand_forecasting": create_demand_dataset(
            source["order_items"], source["orders"], source["products"]
        ),
        "customer_purchase_prediction": create_customer_purchase_dataset(
            source["customers"], source["orders"], source["order_items"], future_orders
        ),
        "product_recommendation": create_recommendation_dataset(
            source["order_items"], source["orders"]
        ),
    }


# ============================================================
# 9. GENERIC ML DATASET CLEANING + VALIDATION
# ============================================================

def clean_ml_dataframe(df, target_columns=None):

    df = df.copy()
    target_columns = target_columns or []

    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_columns = df.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        if col in target_columns:
            continue
        df[col] = df[col].fillna(df[col].median())

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        df[col] = df[col].fillna("Unknown")

    return df.reset_index(drop=True)


def validate_ml_dataframe(df, name, target_columns=None):

    print("\n" + "=" * 60)
    print(f"ML DATASET CHECK: {name}")
    print("=" * 60)

    target_columns = target_columns or []

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nMissing values:")
        print(missing.sort_values(ascending=False))

    print("\nDuplicate rows:", df.duplicated().sum())

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant:
        print("Constant columns:", constant)

    for target in target_columns:
        if target in df.columns:
            print(f"\nTarget distribution ({target}):")
            print(df[target].value_counts(dropna=False))

    print("\nObject columns:", df.select_dtypes(include="object").columns.tolist())
    print("Datetime columns:", df.select_dtypes(include="datetime").columns.tolist())


def clean_all_ml_training_datasets(final_ml_datasets):

    cleaned = {}

    for name, df in final_ml_datasets.items():
        print(f"\nCleaning {name}")
        target_columns = [PRIMARY_TARGET[name]] if name in PRIMARY_TARGET else []
        cleaned[name] = clean_ml_dataframe(df, target_columns=target_columns)
        validate_ml_dataframe(cleaned[name], name, target_columns=target_columns)

    return cleaned


# ============================================================
# 10. PREPROCESSING FRAMEWORK (generic, dataset-agnostic building blocks)
# ============================================================

def drop_selected_columns(df, drop_columns=None):
    df = df.copy()
    auto_drop = []
    for col in df.columns:
        if col.endswith("_id") or col.endswith("_key") or col in ["customer_unique_id"]:
            auto_drop.append(col)

    if drop_columns:
        auto_drop.extend([c for c in drop_columns if c in df.columns])

    auto_drop = list(set(auto_drop))
    df.drop(columns=auto_drop, inplace=True, errors="ignore")
    return df, auto_drop


def process_date_columns(df, date_columns=None):
    df = df.copy()
    if date_columns is None:
        return df

    for col in date_columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_weekday"] = df[col].dt.weekday
        df[f"{col}_quarter"] = df[col].dt.quarter
        df.drop(columns=[col], inplace=True)

    return df


def apply_log_transform(df, columns=None):
    df = df.copy()
    if columns is None:
        return df
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def apply_clipping(df, columns=None, clipping_values=None, fit=True):
    df = df.copy()
    if columns is None:
        return df, clipping_values

    if clipping_values is None:
        clipping_values = {}

    for col in columns:
        if col not in df.columns:
            continue
        if fit:
            clipping_values[col] = {
                "low": df[col].quantile(.01),
                "high": df[col].quantile(.99)
            }
        df[col] = df[col].clip(clipping_values[col]["low"], clipping_values[col]["high"])

    return df, clipping_values


def encode_categories(df, categorical_columns, encoder=None, fit=True):
    df = df.copy()
    if categorical_columns is None:
        return df, encoder

    cols = [c for c in categorical_columns if c in df.columns]
    if len(cols) == 0:
        return df, encoder

    if fit:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cols] = encoder.fit_transform(df[cols].fillna("Unknown").astype(str))
    else:
        df[cols] = encoder.transform(df[cols].fillna("Unknown").astype(str))

    return df, encoder


def handle_missing_values(df, fill_values=None, fit=True):
    df = df.copy()
    if fill_values is None:
        fill_values = {}

    numeric = df.select_dtypes(include=np.number).columns
    for col in numeric:
        if fit:
            fill_values[col] = df[col].median()
        df[col] = df[col].fillna(fill_values[col])

    return df, fill_values


def scale_features(df, scaler=None, fit=True):
    df = df.copy()
    numeric = df.select_dtypes(include=np.number).columns
    if len(numeric) == 0:
        return df, scaler

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[numeric] = scaler.fit_transform(df[numeric])
    else:
        df[numeric] = scaler.transform(df[numeric])

    return df, scaler


def preprocess_data(
    df,
    date_columns=None,
    categorical_columns=None,
    drop_columns=None,
    log_columns=None,
    clip_columns=None,
    scaling="standard",
    encoder=None,
    scaler=None,
    fill_values=None,
    clipping_values=None,
    fit=True
):
    df = df.copy()

    df, dropped = drop_selected_columns(df, drop_columns)
    df = process_date_columns(df, date_columns)
    df = apply_log_transform(df, log_columns)
    df, clipping_values = apply_clipping(df, clip_columns, clipping_values, fit)
    df, encoder = encode_categories(df, categorical_columns, encoder, fit)
    df, fill_values = handle_missing_values(df, fill_values, fit)

    if scaling == "standard":
        df, scaler = scale_features(df, scaler, fit)

    return {
        "data": df,
        "encoder": encoder,
        "scaler": scaler,
        "fill_values": fill_values,
        "clipping_values": clipping_values,
        "dropped_columns": dropped
    }


def preprocess_all_ml_datasets(final_ml_clean):
    """
    Generic, config-driven preprocessing over every entry in
    PREPROCESS_CONFIG. Every dataset's target column is set aside
    before preprocessing (so it's never scaled/clipped/encoded by
    accident), then reattached afterward using the original row index -
    preprocess_data never drops or reorders rows, so this alignment is
    always safe.
    """

    processed = {}
    artifacts = {}

    for name, config in PREPROCESS_CONFIG.items():

        print("\n" + "=" * 80)
        print(name.upper())
        print("=" * 80)

        df = final_ml_clean[name]
        target = PRIMARY_TARGET[name]

        features_only = df.drop(columns=[target])
        artifact = preprocess_data(df=features_only, scaling="standard", fit=True, **config)

        combined = artifact["data"].copy()
        combined[target] = df.loc[combined.index, target]

        print("Final shape:", combined.shape)
        bad_columns = combined.select_dtypes(include=["object", "category", "datetime"]).columns.tolist()
        # the protected time column (if any) is expected to still be datetime here
        protected_time_col = TIME_COLUMNS.get(name)
        bad_columns = [c for c in bad_columns if c != protected_time_col]
        if bad_columns:
            print("WARNING - non-numeric columns remain:", bad_columns)
        else:
            print("OK - all features numeric (plus protected target/time columns)")

        processed[name] = combined
        artifacts[name] = artifact

    return processed, artifacts


# ============================================================
# 11. TRAIN / TEST SPLITS
# ============================================================

def split_ml_dataset(
    df,
    target_column,
    max_rows=50000,
    split_type="random",
    test_size=0.2,
    group_column=None,
    time_column=None,
    random_state=42
):
    """
    split_type: "random" | "time" | "group"
    time_column / group_column are used for ordering/grouping only and
    are dropped from the returned feature matrices - they are never fed
    to the model as raw features.
    """

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in dataframe")

    if split_type not in ["random", "time", "group"]:
        raise ValueError("split_type must be random, time, or group")

    if len(df) > max_rows:

        if split_type == "time":
            df = df.sort_values(time_column)

        sample_df = df.iloc[:max_rows].copy()
        unused_df = df.iloc[max_rows:].copy()

    else:
        sample_df = df.copy()
        unused_df = pd.DataFrame()

    X = sample_df.drop(columns=[target_column])
    y = sample_df[target_column]

    if split_type == "random":

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    elif split_type == "time":

        if time_column is None:
            raise ValueError("time_column required for time split")

        ordered_df = sample_df.sort_values(time_column)
        split_pos = int(len(ordered_df) * (1 - test_size))

        train_df = ordered_df.iloc[:split_pos]
        test_df = ordered_df.iloc[split_pos:]

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

    elif split_type == "group":

        if group_column is None:
            raise ValueError("group_column required for group split")

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_index, test_index = next(splitter.split(X, y, groups=sample_df[group_column]))

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

    drop_extra = [c for c in [time_column, group_column] if c]
    if drop_extra:
        X_train = X_train.drop(columns=[c for c in drop_extra if c in X_train.columns], errors="ignore")
        X_test = X_test.drop(columns=[c for c in drop_extra if c in X_test.columns], errors="ignore")

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    unused_df = unused_df.reset_index(drop=True)

    print("Training rows:", len(X_train), "| Testing rows:", len(X_test),
          "| Unused rows:", len(unused_df), "| Features:", X_train.shape[1])

    return X_train, X_test, y_train, y_test, unused_df


def split_all_ml_datasets(processed_ml_datasets, max_rows=LEARNING_SAMPLE_SIZE):

    splits = {}

    for name, target in PRIMARY_TARGET.items():

        print(f"\nSplitting: {name} (target = {target})")

        X_train, X_test, y_train, y_test, _ = split_ml_dataset(
            processed_ml_datasets[name],
            target_column=target,
            max_rows=max_rows,
            test_size=0.20,
            random_state=RANDOM_STATE,
            **SPLIT_CONFIG[name]
        )

        splits[name] = (X_train, X_test, y_train, y_test)

    return splits


# ============================================================
# 12. RECOMMENDATION MATRIX (unsupervised - handled separately since
# it has no target and isn't part of PRIMARY_TARGET/split_datasets)
# ============================================================

def build_recommendation_matrix(recommendation_df):
    customer_codes = recommendation_df["customer_id"].astype("category").cat.codes
    product_codes = recommendation_df["product_id"].astype("category").cat.codes
    return csr_matrix((recommendation_df["purchase_count"], (customer_codes, product_codes)))


# ============================================================
# 13. MODEL REGISTRY
# ============================================================

def build_model_registry():
    """
    Factory that returns a FRESH model registry every time it's called.
    Kept as a factory (rather than a shared module-level dict) so every
    pipeline run / retrain gets its own model instances instead of
    accidentally sharing (and mutating, e.g. via set_params) the same
    objects across runs or across multiple app contexts.
    """

    return {

        "delivery_delay": {
            "task": "classification",
            "time_series": False,
            "scoring": "roc_auc",
            "model": LGBMClassifier(
                objective="binary", random_state=RANDOM_STATE, n_jobs=DEFAULT_N_JOBS, verbosity=-1
            ),
            "params": {
                "n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6],
                "num_leaves": [20, 40], "min_child_samples": [20, 50], "scale_pos_weight": [1, 3]
            }
        },

        "order_cancellation": {
            "task": "classification",
            "time_series": False,
            "scoring": "average_precision",
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=DEFAULT_N_JOBS,
                max_samples=0.75, class_weight="balanced_subsample"
            ),
            "params": {
                "n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_leaf": [2, 5],
                "max_features": ["sqrt"], "class_weight": ["balanced", "balanced_subsample"]
            }
        },

        "review_prediction": {
            "task": "regression",
            "time_series": False,
            "scoring": "neg_mean_absolute_error",
            "model": ElasticNet(
                alpha=0.1, selection="random", random_state=RANDOM_STATE, max_iter=5000, tol=0.001
            ),
            "params": {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]}
        },

        "demand_forecasting": {
            "task": "regression",
            "time_series": True,
            "scoring": "neg_root_mean_squared_error",
            "model": XGBRegressor(
                objective="reg:squarederror", random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS, tree_method="hist"
            ),
            "params": {
                "n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6],
                "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]
            }
        },

        "customer_purchase_prediction": {
            "task": "classification",
            "time_series": False,
            "scoring": "roc_auc",
            "model": XGBClassifier(
                objective="binary:logistic", random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS, tree_method="hist", eval_metric="logloss"
            ),
            "params": {
                "n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6],
                "scale_pos_weight": [1, 3]
            }
        },

        "product_recommendation": {
            "task": "unsupervised",
            "time_series": False,
            "scoring": None,
            "model": NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=DEFAULT_N_JOBS),
            "params": {"n_neighbors": [5, 10, 20], "metric": ["cosine"], "algorithm": ["auto", "brute"]}
        },
    }


def validate_model_registry(models):

    print("\n" + "=" * 60)
    print("MODEL REGISTRY VALIDATION")
    print("=" * 60)

    required_keys = ["task", "model", "params"]

    for name, info in models.items():
        missing = [key for key in required_keys if key not in info]
        print(f"{name}: {'Missing ' + str(missing) if missing else 'READY'}")


def compute_scale_pos_weight(y):
    """
    Class-imbalance helper: ratio of negative to positive labels.
    Returns 1.0 if there are no positives (nothing to weight against).
    """

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    return negatives / positives if positives > 0 else 1.0


# ============================================================
# 14. MODEL TUNING FUNCTION
# ============================================================

def tune_model(model_name, X_train, y_train, models, search_type="random", iterations=10, cv_splits=3):

    print("\n" + "=" * 80)
    print(f"TUNING MODEL: {model_name}")
    print("=" * 80)

    if model_name not in models:
        raise ValueError(f"{model_name} not found in registry")

    config = models[model_name]
    task = config["task"]

    if task == "unsupervised":
        raise ValueError("Unsupervised models do not use the tuning wrapper")

    model = config["model"]
    params = config["params"]
    time_series = config.get("time_series", False)
    scoring = config["scoring"]

    print(f"Training rows: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    if task == "classification":
        print("Target distribution:")
        print(y_train.value_counts())

        if y_train.nunique() < 2:
            print(f"{model_name}: target has only one class - skipping tuning.")
            return None

    if time_series:
        cv = TimeSeriesSplit(n_splits=cv_splits)
    elif task == "classification" and y_train.nunique() == 2:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = cv_splits

    start_time = time.time()

    # search itself stays single-threaded on purpose - the underlying
    # models already parallelize via DEFAULT_N_JOBS, and nesting two
    # levels of parallelism is what actually kills laptop performance.
    if search_type == "grid":
        search = GridSearchCV(
            estimator=model, param_grid=params, scoring=scoring,
            cv=cv, n_jobs=1, error_score="raise", return_train_score=True
        )
    else:
        search = RandomizedSearchCV(
            estimator=model, param_distributions=params, n_iter=min(iterations, 10),
            scoring=scoring, cv=cv, random_state=RANDOM_STATE, n_jobs=1,
            error_score="raise", return_train_score=True
        )

    try:
        search.fit(X_train, y_train)
    except Exception as e:
        print("\nMODEL FAILED")
        print("Error type:", type(e).__name__)
        print("Reason:", e)
        return None

    elapsed = time.time() - start_time

    print(f"\nRuntime: {elapsed:.2f} seconds")
    print("Best score:", search.best_score_)
    print("Best parameters:", search.best_params_)

    return {
        "model": search.best_estimator_,
        "parameters": search.best_params_,
        "score": search.best_score_,
        "runtime_seconds": elapsed
    }


# ============================================================
# 15. FINAL MODEL TRAINING + TEST EVALUATION
# ============================================================

def train_model(tuned_result, X_train, X_test, y_train, y_test, task):

    print("\n" + "=" * 80)
    print("FINAL MODEL TRAINING")
    print("=" * 80)

    start_time = time.time()

    model = tuned_result["model"]
    parameters = tuned_result["parameters"]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = {}

    if task == "classification":

        results["accuracy"] = accuracy_score(y_test, predictions)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
            results["roc_auc"] = roc_auc_score(y_test, probabilities)

    else:

        results["rmse"] = mean_squared_error(y_test, predictions) ** 0.5
        results["mae"] = mean_absolute_error(y_test, predictions)
        results["r2"] = r2_score(y_test, predictions)

    runtime = time.time() - start_time

    print("\nTEST RESULTS")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")

    return {"model": model, "parameters": parameters, "score": results, "runtime_seconds": runtime}


def train_recommendation_model(matrix, models, model_name="product_recommendation"):

    print("\n" + "=" * 80)
    print("TRAINING RECOMMENDATION MODEL")
    print("=" * 80)

    if model_name not in models:
        raise ValueError(f"{model_name} not found in registry")

    if models[model_name]["task"] != "unsupervised":
        raise ValueError("This function is only for unsupervised models")

    start_time = time.time()

    model = NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=DEFAULT_N_JOBS)
    model.fit(matrix)

    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.2f} seconds")

    return {
        "model": model,
        "parameters": {"n_neighbors": 5, "metric": "cosine"},
        "score": None,
        "runtime_seconds": runtime
    }


def create_model_package(model, parameters, score, preprocessing_artifact=None,
                          target_name=None, model_name=None):
    """Bundles a trained model with its parameters, score, and (optionally)
    the preprocessing artifact needed to transform new raw rows the same
    way at inference time."""

    package = {
        "model": model,
        "parameters": parameters,
        "score": score,
        "model_name": model_name,
        "target": target_name,
        "created": str(datetime.now()),
        "encoder": None,
        "scaler": None,
        "fill_values": None,
        "clipping_values": None,
        "dropped_columns": None,
        "feature_names": None,
    }

    if preprocessing_artifact is not None:
        package["encoder"] = preprocessing_artifact.get("encoder")
        package["scaler"] = preprocessing_artifact.get("scaler")
        package["fill_values"] = preprocessing_artifact.get("fill_values")
        package["clipping_values"] = preprocessing_artifact.get("clipping_values")
        package["dropped_columns"] = preprocessing_artifact.get("dropped_columns")
        package["feature_names"] = preprocessing_artifact["data"].columns.tolist()

    return package


# ============================================================
# 16. PICKLE SAVE / LOAD + EXPORT / IMPORT WRAPPERS
# ============================================================

def save_pickle(obj, path):
    """Save a Python object to a pickle file."""

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved artifact: {path}")


def load_pickle(path):
    """Load a Python object from a pickle file."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "rb") as file:
        obj = pickle.load(file)

    print(f"Loaded artifact: {path}")
    return obj


def export_model_package(package, folder, filename):

    os.makedirs(folder, exist_ok=True)
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    filepath = os.path.join(folder, filename)
    save_pickle(package, filepath)
    return filepath


def import_model_package(folder, filename):

    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model package not found: {filepath}")

    return load_pickle(filepath)


# ============================================================
# 17. MODEL MONITORING + SELF-ADJUSTING RETRAINING
# ============================================================

def monitor_and_retrain_model(
    model_name,
    current_package,
    X_train,
    X_test,
    y_train,
    y_test,
    models,
    performance_threshold=0.0,
    save_path=None
):
    """
    Evaluates the currently deployed model against a freshly-tuned
    candidate trained on the latest data, and only swaps the model if
    the candidate is actually better (>= performance_threshold
    improvement). Otherwise the current model is kept as-is.
    """

    print("\n" + "=" * 80)
    print(f"MODEL MONITORING + RETRAINING: {model_name}")
    print("=" * 80)

    task = models[model_name]["task"]
    current_model = current_package["model"]

    current_predictions = current_model.predict(X_test)

    if task == "classification":
        current_probabilities = current_model.predict_proba(X_test)[:, 1]
        current_score = roc_auc_score(y_test, current_probabilities)
    else:
        current_score = r2_score(y_test, current_predictions)

    print("Current model score:", current_score)

    if task == "classification":
        scale_pos_weight = compute_scale_pos_weight(y_train)
        print("Dynamic class weight (scale_pos_weight):", scale_pos_weight)

        model_object = models[model_name]["model"]
        if "scale_pos_weight" in model_object.get_params():
            model_object.set_params(scale_pos_weight=scale_pos_weight)

    print("\nTraining candidate model...")
    tuned_result = tune_model(model_name, X_train, y_train, models)

    if tuned_result is None:
        print("Candidate failed to tune - current model kept")
        return current_package

    candidate_result = train_model(tuned_result, X_train, X_test, y_train, y_test, task)

    candidate_score = (
        candidate_result["score"]["roc_auc"] if task == "classification"
        else candidate_result["score"]["r2"]
    )

    print("Candidate score:", candidate_score)
    improvement = candidate_score - current_score
    print("Improvement:", improvement)

    if improvement >= performance_threshold:

        print("\nNew model accepted")
        package = create_model_package(
            candidate_result["model"], candidate_result["parameters"], candidate_result["score"],
            model_name=model_name, target_name=PRIMARY_TARGET.get(model_name)
        )
        if save_path:
            save_pickle(package, save_path)
        return package

    print("\nCurrent model kept")
    return current_package


# ============================================================
# 18. TRAIN / INGEST ORCHESTRATION
# ============================================================

def train_all_initial_models(split_datasets, models_registry, preprocess_artifacts,
                              recommendation_matrix, save_dir=MODEL_SAVE_DIR):
    """
    Trains every supervised model once on split_datasets, plus the
    unsupervised recommendation model (if there's any interaction data).
    Saves each trained package to disk and returns a dict of them.
    """

    os.makedirs(save_dir, exist_ok=True)
    trained_packages = {}

    for name, (X_train, X_test, y_train, y_test) in split_datasets.items():

        task = models_registry[name]["task"]

        if task == "classification":

            if y_train.nunique() < 2:
                print(f"{name} skipped: target has only one class "
                      f"(try a larger LEARNING_SAMPLE_SIZE or a longer future window).")
                continue

            scale_pos_weight = compute_scale_pos_weight(y_train)
            print(f"[{name}] scale_pos_weight set to {scale_pos_weight:.3f} "
                  f"(positives={int((y_train == 1).sum())}, negatives={int((y_train == 0).sum())})")

            model_object = models_registry[name]["model"]
            if "scale_pos_weight" in model_object.get_params():
                model_object.set_params(scale_pos_weight=scale_pos_weight)

        tuned = tune_model(name, X_train, y_train, models_registry)

        if tuned is None:
            print(f"{name} failed during tuning - no model saved.")
            continue

        result = train_model(tuned, X_train, X_test, y_train, y_test, task)

        package = create_model_package(
            result["model"], result["parameters"], result["score"],
            preprocessing_artifact=preprocess_artifacts.get(name),
            target_name=PRIMARY_TARGET[name], model_name=name
        )
        trained_packages[name] = package

        export_model_package(package, folder=save_dir, filename=name)

    if recommendation_matrix is not None:

        recommendation_result = train_recommendation_model(recommendation_matrix, models_registry)
        trained_packages["product_recommendation"] = create_model_package(
            recommendation_result["model"], recommendation_result["parameters"], recommendation_result["score"],
            model_name="product_recommendation"
        )
        export_model_package(
            trained_packages["product_recommendation"], folder=save_dir, filename="product_recommendation"
        )
    else:
        print("Recommendation skipped: no interaction data in the learning sample")

    return trained_packages


def ingest_new_data_and_retrain(
    new_raw_tables,
    context,
    future_orders=None,
    max_rows=None,
    save_dir=None,
    export_row_usage=EXPORT_ROW_USAGE_MANIFEST,
    row_usage_folder=None,
):
    """
    Self-adjusting training entry point, rewritten to take an explicit
    `context` dict (as produced by run_initial_pipeline(), or by a
    previous call to this function) instead of module-level globals.

    `new_raw_tables` is a dict that may contain any subset of the keys
    {"orders", "customers", "order_items", "products", "sellers",
    "payments", "reviews"} (same columns as the corresponding raw
    tables). It:

      1. appends the new rows to the matching historical raw table,
      2. samples a fresh learning sample from the combined pool
         (rebuild_related_tables keeps everything relationally
         consistent),
      3. rebuilds every engineered feature table, re-cleans,
         re-preprocesses, and re-splits it,
      4. asks monitor_and_retrain_model to decide - per model - whether
         a freshly tuned candidate actually beats the currently deployed
         model before replacing it,
      5. refreshes the (unsupervised) recommendation model directly,
         since it has no target/score to compare against,
      6. rebuilds and (optionally) re-exports the row-usage manifest.

    Models that don't improve are left untouched, so a batch of noisy or
    low-volume new data can never silently degrade production models -
    only genuine improvements get deployed.

    Returns a NEW context dict - store this back wherever you keep your
    application's pipeline state (do not keep using the old one).
    """

    max_rows = max_rows or context.get("sample_size", LEARNING_SAMPLE_SIZE)
    save_dir = save_dir or context.get("model_save_dir", MODEL_SAVE_DIR)
    row_usage_folder = row_usage_folder or ROW_USAGE_FOLDER

    print("\n" + "=" * 80)
    print("INGESTING NEW DATA AND RE-EVALUATING MODELS")
    print("=" * 80)

    tables = dict(context["tables"])
    trained_packages = dict(context["trained_packages"])
    models_registry = context["models"]

    for key, new_df in new_raw_tables.items():

        if key not in tables:
            print(f"Skipping unknown table key: {key}")
            continue

        new_clean = clean_dataframe(new_df)
        current = tables[key]
        updated = pd.concat([current, new_clean], ignore_index=True).drop_duplicates().reset_index(drop=True)
        tables[key] = updated
        print(f"Updated {key}: {len(current):,} -> {len(updated):,} rows")

    if future_orders is None:
        future_orders = context["future_source"]["orders"]

    all_orders = tables["orders"].sort_values("order_purchase_timestamp").reset_index(drop=True)
    n_learn = min(max_rows, len(all_orders))
    refreshed_learning_orders = all_orders.sample(n=n_learn, random_state=RANDOM_STATE)

    refreshed_learning_source = rebuild_related_tables(refreshed_learning_orders, tables)
    validate_star_schema(refreshed_learning_source, label="REFRESHED LEARNING SAMPLE")

    refreshed_ml_data = create_all_ml_feature_tables(refreshed_learning_source, future_orders)
    refreshed_ml_clean = clean_all_ml_training_datasets(refreshed_ml_data)
    refreshed_processed, refreshed_artifacts = preprocess_all_ml_datasets(refreshed_ml_clean)
    refreshed_splits = split_all_ml_datasets(refreshed_processed, max_rows=max_rows)

    for name, (X_train, X_test, y_train, y_test) in refreshed_splits.items():

        if name not in trained_packages:
            continue

        trained_packages[name] = monitor_and_retrain_model(
            model_name=name,
            current_package=trained_packages[name],
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            models=models_registry,
            save_path=os.path.join(save_dir, f"{name}.pkl")
        )

    refreshed_matrix = (
        build_recommendation_matrix(refreshed_ml_clean["product_recommendation"])
        if len(refreshed_ml_clean["product_recommendation"]) > 0 else None
    )

    if refreshed_matrix is not None:
        recommendation_result = train_recommendation_model(refreshed_matrix, models_registry)
        trained_packages["product_recommendation"] = create_model_package(
            recommendation_result["model"], recommendation_result["parameters"], recommendation_result["score"],
            model_name="product_recommendation"
        )
        export_model_package(
            trained_packages["product_recommendation"], folder=save_dir, filename="product_recommendation"
        )

    row_usage_manifest = build_row_usage_manifest(refreshed_learning_source, tables)
    if export_row_usage:
        export_row_usage_manifest(row_usage_manifest, output_folder=row_usage_folder)

    new_context = dict(context)
    new_context.update({
        "tables": tables,
        "learning_orders": refreshed_learning_orders,
        "learning_source": refreshed_learning_source,
        "row_usage_manifest": row_usage_manifest,
        "ml_data": refreshed_ml_data,
        "ml_data_clean": refreshed_ml_clean,
        "processed_ml_datasets": refreshed_processed,
        "preprocess_artifacts": refreshed_artifacts,
        "split_datasets": refreshed_splits,
        "recommendation_matrix": refreshed_matrix,
        "trained_packages": trained_packages,
        "sample_size": max_rows,
        "model_save_dir": save_dir,
    })

    return new_context


# ============================================================
# 19. OPTIONAL: EXPORT FINAL ML FEATURE TABLES
# ============================================================

def export_ml_feature_tables(ml_data, folder=ML_TRAINING_FOLDER):
    """
    Writes each engineered per-model feature table (not raw source
    rows) to CSV. Distinct from the row-usage manifest above - this is
    just a convenience dump of the final modeling inputs.
    """

    os.makedirs(folder, exist_ok=True)

    for name, df in ml_data.items():
        df.to_csv(f"{folder}/{name}.csv", index=False)

    print(f"ML training tables written to ./{folder}/")


# ============================================================
# TOP-LEVEL ORCHESTRATOR - RUN THE WHOLE PIPELINE ONCE
# ============================================================

def run_initial_pipeline(
    dataset_slug=DATASET_SLUG,
    sample_size=LEARNING_SAMPLE_SIZE,
    model_save_dir=MODEL_SAVE_DIR,
    row_usage_folder=ROW_USAGE_FOLDER,
    ml_training_folder=ML_TRAINING_FOLDER,
    train_fraction=0.70,
    cleanup_first=True,
    export_row_usage=EXPORT_ROW_USAGE_MANIFEST,
    export_ml_tables=True,
):
    """
    Runs the whole pipeline end-to-end (download -> clean -> validate ->
    time-split -> sample -> engineer features -> preprocess -> split ->
    train every model -> build the row-usage manifest) and returns a
    single `context` dict holding every intermediate artifact.

    Keep this context around (in memory, in your app's session/state
    store, pickled to disk - whatever fits) and pass it straight into
    ingest_new_data_and_retrain() whenever new raw Olist rows arrive.
    """

    if cleanup_first:
        cleanup_previous_run(dataset_slug)

    raw_datasets = download_and_load_datasets(dataset_slug)
    raw_frames = extract_required_frames(raw_datasets)
    tables = clean_all_raw_tables(raw_frames)

    validate_star_schema(tables, label="FULL CLEANED DATA")

    orders_train_full, orders_future, train_end_date = split_orders_by_time(
        tables["orders"], train_fraction=train_fraction
    )
    print("\nTRAIN END DATE:", train_end_date)

    future_source = rebuild_related_tables(orders_future, tables)
    validate_star_schema(future_source, label="FUTURE SOURCE (held out, unsampled)")

    learning_orders, rest_orders = sample_learning_orders(
        orders_train_full, orders_future, sample_size=sample_size
    )

    learning_source = rebuild_related_tables(learning_orders, tables)
    rest_source = rebuild_related_tables(rest_orders, tables)

    validate_star_schema(learning_source, label="LEARNING SAMPLE")
    validate_star_schema(rest_source, label="REST DATA")

    print(f"\nLearning sample orders: {len(learning_orders):,}")
    print(f"Rest data orders:       {len(rest_orders):,}")

    row_usage_manifest = build_row_usage_manifest(learning_source, tables)
    if export_row_usage:
        export_row_usage_manifest(row_usage_manifest, output_folder=row_usage_folder)

    ml_data = create_all_ml_feature_tables(learning_source, future_source["orders"])

    print("\nML DATASETS CREATED")
    for name, df in ml_data.items():
        print(f"{name}: {df.shape} | "
              f"memory MB: {round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)} | "
              f"missing: {df.isna().sum().sum()}")

    ml_data_clean = clean_all_ml_training_datasets(ml_data)

    processed_ml_datasets, preprocess_artifacts = preprocess_all_ml_datasets(ml_data_clean)

    split_datasets = split_all_ml_datasets(processed_ml_datasets, max_rows=sample_size)

    print("\n" + "=" * 80)
    print("TRAIN / TEST SUMMARY")
    print("=" * 80)
    for name, (Xtr, Xte, ytr, yte) in split_datasets.items():
        print(f"{name:32s} X_train {Xtr.shape} | X_test {Xte.shape}")

    recommendation_matrix = (
        build_recommendation_matrix(ml_data_clean["product_recommendation"])
        if len(ml_data_clean["product_recommendation"]) > 0 else None
    )
    print("\nRecommendation matrix shape:",
          recommendation_matrix.shape if recommendation_matrix is not None else "N/A (no interaction data)")

    models_registry = build_model_registry()
    validate_model_registry(models_registry)

    trained_packages = train_all_initial_models(
        split_datasets, models_registry, preprocess_artifacts, recommendation_matrix,
        save_dir=model_save_dir
    )

    if export_ml_tables:
        export_ml_feature_tables(ml_data, folder=ml_training_folder)

    context = {
        "dataset_slug": dataset_slug,
        "tables": tables,
        "train_end_date": train_end_date,
        "orders_train_full": orders_train_full,
        "orders_future": orders_future,
        "future_source": future_source,
        "learning_orders": learning_orders,
        "rest_orders": rest_orders,
        "learning_source": learning_source,
        "rest_source": rest_source,
        "row_usage_manifest": row_usage_manifest,
        "ml_data": ml_data,
        "ml_data_clean": ml_data_clean,
        "processed_ml_datasets": processed_ml_datasets,
        "preprocess_artifacts": preprocess_artifacts,
        "split_datasets": split_datasets,
        "recommendation_matrix": recommendation_matrix,
        "models": models_registry,
        "trained_packages": trained_packages,
        "sample_size": sample_size,
        "model_save_dir": model_save_dir,
    }

    print("\nPipeline complete.")

    return context




pipeline_context = run_initial_pipeline()

print("\n" + "=" * 80)
print("OLIST PIPELINE COMPLETE")
print("=" * 80)
print(f"Learning sample orders used for training: {len(pipeline_context['learning_orders']):,}")
print(f"Rest data set aside (not touched by modeling): {len(pipeline_context['rest_orders']):,}")
print(f"Row usage manifest:     ./{ROW_USAGE_FOLDER}/")
print(f"ML training tables:     ./{ML_TRAINING_FOLDER}/")
print(f"Saved models:           ./{MODEL_SAVE_DIR}/")
for name, package in pipeline_context["trained_packages"].items():
    print(f"  {name:32s} -> score: {package['score']}")

print(
    "\nTo self-adjust as new raw data comes in later, call:\n"
    "    pipeline_context = ingest_new_data_and_retrain(new_raw_tables, pipeline_context)\n"
    "where new_raw_tables is a dict like {'orders': new_orders_df, "
    "'order_items': new_items_df, ...}. Each model is only replaced if the "
    "retrained candidate actually beats the currently deployed one on held-out data. "
    f"Every call also refreshes the row-usage manifest under '{ROW_USAGE_FOLDER}/' so "
    "you always know exactly which source rows (per table) have been consumed so far."
)