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
