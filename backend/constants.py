from pathlib import Path

# Root
ROOT_DIR = Path(__file__).resolve().parent.parent

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

export_row_usage_manifest = True