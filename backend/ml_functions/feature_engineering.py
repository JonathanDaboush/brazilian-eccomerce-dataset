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