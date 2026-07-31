from constants import *
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import (

    train_test_split,
    GroupShuffleSplit
)

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