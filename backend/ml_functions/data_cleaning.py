import numpy as np
import pandas as pd
from constants import *

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
