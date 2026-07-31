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