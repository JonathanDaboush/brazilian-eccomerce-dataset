import numpy as np
import pandas as pd
import os
import glob
#import kagglehub


# ============================================================
# 1. EXTRACT: DOWNLOAD AND LOAD DATASETS
# ============================================================

#path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
path="#"
csv_files = glob.glob(
    os.path.join(path, "**", "*.csv"),
    recursive=True
)


datasets = {}

for file in csv_files:

    file_name = os.path.splitext(
        os.path.basename(file)
    )[0]

    datasets[file_name] = pd.read_csv(file)


print("LOADED DATASETS\n")

for name, df in datasets.items():
    print(
        f"{name}: {df.shape}"
    )



# ============================================================
# 2. CREATE DATAFRAME REFERENCES
# ============================================================

df_customers = datasets['olist_customers_dataset']
df_sellers = datasets['olist_sellers_dataset']
df_order_reviews = datasets['olist_order_reviews_dataset']
df_order_items = datasets['olist_order_items_dataset']
df_products = datasets['olist_products_dataset']
df_geolocation = datasets['olist_geolocation_dataset']
df_category_trans = datasets['product_category_name_translation']
df_orders = datasets['olist_orders_dataset']
df_order_payments = datasets['olist_order_payments_dataset']



# ============================================================
# 3. DATATYPE CLEANING
# ============================================================


order_date_cols = [

    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'

]


for col in order_date_cols:

    df_orders[col] = pd.to_datetime(
        df_orders[col],
        errors='coerce'
    )



review_date_cols = [

    'review_creation_date',
    'review_answer_timestamp'

]


for col in review_date_cols:

    df_order_reviews[col] = pd.to_datetime(
        df_order_reviews[col],
        errors='coerce'
    )



# ============================================================
# 4. NULL VALUE CLEANING
# ============================================================


# Reviews text

review_text_cols = [

    'review_comment_title',
    'review_comment_message'

]


for col in review_text_cols:

    df_order_reviews[col] = (
        df_order_reviews[col]
        .fillna('')
    )



# Product category

df_products['product_category_name'] = (
    df_products['product_category_name']
    .fillna('unknown')
)



# Product numerical attributes

product_numeric_cols = [

    'product_name_lenght',
    'product_description_lenght',
    'product_photos_qty',
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm'

]


for col in product_numeric_cols:

    df_products[col] = pd.to_numeric(
        df_products[col],
        errors='coerce'
    )

    df_products[col] = (
        df_products[col]
        .fillna(
            df_products[col].median()
        )
    )



# ============================================================
# 5. REMOVE EXACT DUPLICATES
# ============================================================


print("\nREMOVING EXACT DUPLICATES\n")


for name, df in datasets.items():

    before = len(df)

    df.drop_duplicates(
        inplace=True
    )

    after = len(df)


    if before != after:

        print(
            f"{name}: removed {before-after} duplicates"
        )



# ============================================================
# 6. PRIMARY KEY CLEANING
# ============================================================


df_products = (
    df_products
    .drop_duplicates(
        subset=['product_id'],
        keep='first'
    )
)


df_customers = (
    df_customers
    .drop_duplicates(
        subset=['customer_id'],
        keep='first'
    )
)


df_sellers = (
    df_sellers
    .drop_duplicates(
        subset=['seller_id'],
        keep='first'
    )
)


df_category_trans = (
    df_category_trans
    .drop_duplicates(
        subset=['product_category_name'],
        keep='first'
    )
)



# ============================================================
# 7. CREATE SYNTHETIC PRIMARY KEYS FOR FACT TABLES
# ============================================================


# order item grain
df_order_items['order_item_key'] = range(
    1,
    len(df_order_items)+1
)



# payment grain
df_order_payments['payment_id'] = range(
    1,
    len(df_order_payments)+1
)



# review grain
df_order_reviews['review_key'] = range(
    1,
    len(df_order_reviews)+1
)



# ============================================================
# 8. GEOLOCATION NORMALIZATION
# ============================================================


df_geolocation = (

    df_geolocation

    .groupby(
        'geolocation_zip_code_prefix'
    )

    .agg({

        'geolocation_lat':'mean',
        'geolocation_lng':'mean',
        'geolocation_city':'first',
        'geolocation_state':'first'

    })

    .reset_index()

)



# ============================================================
# 9. REVIEW HANDLING
# KEEP LATEST REVIEW PER ORDER
# ============================================================


df_order_reviews = (

    df_order_reviews

    .sort_values(
        'review_answer_timestamp'
    )

    .drop_duplicates(
        subset=['order_id'],
        keep='last'
    )

)



# ============================================================
# 10. REBUILD DATASET DICTIONARY
# ============================================================


datasets = {

    'customers': df_customers,

    'sellers': df_sellers,

    'orders': df_orders,

    'order_items': df_order_items,

    'products': df_products,

    'payments': df_order_payments,

    'reviews': df_order_reviews,

    'geolocation': df_geolocation,

    'category_translation': df_category_trans

}



# ============================================================
# 11. FINAL DATA QUALITY CHECK
# ============================================================


print("\n==============================")
print("FINAL DATA QUALITY CHECK")
print("==============================")



for name, df in datasets.items():

    print(
        f"\n{name.upper()}"
    )

    print(
        "Shape:",
        df.shape
    )

    print(
        "Nulls:",
        df.isnull().sum().sum()
    )



# ============================================================
# 12. PRIMARY KEY VALIDATION
# ============================================================


print("\n==============================")
print("PRIMARY KEY VALIDATION")
print("==============================")


primary_keys = {

    'customers':'customer_id',

    'sellers':'seller_id',

    'products':'product_id',

    'orders':'order_id',

    'order_items':'order_item_key',

    'payments':'payment_id',

    'reviews':'review_key',

    'geolocation':'geolocation_zip_code_prefix',

    'category_translation':'product_category_name'

}



for table,key in primary_keys.items():

    df = datasets[table]


    print(
        f"\n{table}"
    )

    print(
        "Duplicate PK:",
        df[key].duplicated().sum()
    )


    print(
        "Null PK:",
        df[key].isnull().sum()
    )



# ============================================================
# 13. FOREIGN KEY VALIDATION
# ============================================================


print("\n==============================")
print("FOREIGN KEY VALIDATION")
print("==============================")


checks = {


"orders -> customers":
(
    df_orders['customer_id'],
    df_customers['customer_id']
),


"items -> products":
(
    df_order_items['product_id'],
    df_products['product_id']
),


"items -> sellers":
(
    df_order_items['seller_id'],
    df_sellers['seller_id']
),


"items -> orders":
(
    df_order_items['order_id'],
    df_orders['order_id']
),


"reviews -> orders":
(
    df_order_reviews['order_id'],
    df_orders['order_id']
),


"payments -> orders":
(
    df_order_payments['order_id'],
    df_orders['order_id']
)

}



for name,(child,parent) in checks.items():

    missing = (
        ~child.isin(parent)
    ).sum()


    print(
        name,
        ":",
        missing
    )



# ============================================================
# 14. BUSINESS SANITY CHECKS
# ============================================================


print("\n==============================")
print("BUSINESS SANITY CHECKS")
print("==============================")


print(
    "Negative prices:",
    (df_order_items['price'] < 0).sum()
)


print(
    "Negative payments:",
    (df_order_payments['payment_value'] < 0).sum()
)



invalid_delivery = (

    df_orders['order_delivered_customer_date']

    <

    df_orders['order_purchase_timestamp']

)


print(
    "Invalid delivery dates:",
    invalid_delivery.sum()
)



print("\nPIPELINE READY FOR WAREHOUSE MODELING")

df_customers.to_csv('customers.csv', index=False)
df_sellers.to_csv('sellers.csv', index=False) 
df_order_reviews.to_csv('order_reviews.csv', index=False) 
df_order_items.to_csv('order_items.csv', index=False)
df_products.to_csv('products.csv', index=False)
df_geolocation.to_csv('geolocation.csv', index=False)
df_category_trans.to_csv('category_trans.csv', index=False)
df_orders.to_csv('orders.csv', index=False) 
df_order_payments.to_csv('order_payments.csv', index=False)