import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from constants import *
from backend.ml_functions.pre_process import preprocess_data
from backend.ml_functions.retrain import ingest_new_data_and_retrain

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
def load_pickle(path):
    """Load a Python object from a pickle file."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "rb") as file:
        obj = pickle.load(file)

    print(f"Loaded artifact: {path}")
    return obj

def import_model_package(folder, filename):

    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model package not found: {filepath}")

    return load_pickle(filepath)

def build_one_page_pipeline_dashboard(context, run_metrics=None, title="Olist Pipeline Dashboard"):
    run_metrics = run_metrics or {}

    tables = context.get("tables", {})
    row_usage_manifest = context.get("row_usage_manifest", {})
    ml_data_clean = context.get("ml_data_clean", {})
    processed_ml_datasets = context.get("processed_ml_datasets", {})
    split_datasets = context.get("split_datasets", {})
    preprocess_artifacts = context.get("preprocess_artifacts", {})
    trained_packages = context.get("trained_packages", {})

    raw_rows = pd.DataFrame([
        {
            "table": k,
            "rows": len(v),
            "cols": v.shape[1],
            "nulls": int(v.isna().sum().sum()),
            "dupes": int(v.duplicated().sum())
        }
        for k, v in tables.items()
    ])

    usage_rows = pd.DataFrame([
        {
            "table": k,
            "used": v.get("used_count", 0),
            "unused": v.get("unused_count", 0)
        }
        for k, v in row_usage_manifest.items()
    ])

    clean_rows = pd.DataFrame([
        {
            "dataset": k,
            "rows": len(v),
            "cols": v.shape[1],
            "nulls": int(v.isna().sum().sum()),
            "dupes": int(v.duplicated().sum())
        }
        for k, v in ml_data_clean.items()
    ])

    processed_rows = pd.DataFrame([
        {
            "dataset": k,
            "rows": len(v),
            "cols": v.shape[1],
            "numeric": int(len(v.select_dtypes(include=np.number).columns)),
            "datetime": int(len(v.select_dtypes(include="datetime").columns))
        }
        for k, v in processed_ml_datasets.items()
    ])

    split_rows = []
    for name, split in split_datasets.items():
        X_train, X_test, y_train, y_test = split
        split_rows.append({
            "dataset": name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features": X_train.shape[1]
        })
    split_df = pd.DataFrame(split_rows)

    model_rows = []
    for name, pkg in trained_packages.items():
        score = pkg.get("score", {})
        metric = (
            score.get("accuracy")
            if "accuracy" in score else
            score.get("roc_auc")
            if "roc_auc" in score else
            score.get("r2")
            if "r2" in score else
            score.get("rmse")
        )
        model_rows.append({
            "model": name,
            "runtime": pkg.get("runtime_seconds", np.nan),
            "score": metric
        })
    model_df = pd.DataFrame(model_rows)

    improve_rows = []
    for name, m in run_metrics.get("improvement", {}).items():
        improve_rows.append({
            "model": name,
            "baseline": m.get("baseline", np.nan),
            "current": m.get("current", np.nan),
            "delta": m.get("delta", np.nan),
            "accepted": m.get("accepted", False)
        })
    improve_df = pd.DataFrame(improve_rows)

    health_rows = []
    for name, m in run_metrics.get("health", {}).items():
        health_rows.append({
            "metric": name,
            "value": m
        })
    health_df = pd.DataFrame(health_rows)

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "Raw Tables",
            "Row Usage",
            "Cleaned ML Data",
            "Processed Features",
            "Train/Test Split",
            "Model Runtime & Score",
            "Improvement / Retrain Delta",
            "Diagnostics"
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08
    )

    if not raw_rows.empty:
        fig.add_trace(go.Bar(x=raw_rows["table"], y=raw_rows["rows"], name="Rows"), row=1, col=1)
        fig.add_trace(go.Bar(x=raw_rows["table"], y=raw_rows["nulls"], name="Nulls"), row=1, col=1)

    if not usage_rows.empty:
        fig.add_trace(go.Bar(x=usage_rows["table"], y=usage_rows["used"], name="Used"), row=1, col=2)
        fig.add_trace(go.Bar(x=usage_rows["table"], y=usage_rows["unused"], name="Unused"), row=1, col=2)

    if not clean_rows.empty:
        fig.add_trace(go.Bar(x=clean_rows["dataset"], y=clean_rows["nulls"], name="Nulls"), row=2, col=1)
        fig.add_trace(go.Bar(x=clean_rows["dataset"], y=clean_rows["dupes"], name="Dupes"), row=2, col=1)

    if not processed_rows.empty:
        fig.add_trace(go.Bar(x=processed_rows["dataset"], y=processed_rows["cols"], name="Cols"), row=2, col=2)
        fig.add_trace(go.Scatter(x=processed_rows["dataset"], y=processed_rows["numeric"], mode="lines+markers", name="Numeric"), row=2, col=2)

    if not split_df.empty:
        fig.add_trace(go.Bar(x=split_df["dataset"], y=split_df["train_rows"], name="Train"), row=3, col=1)
        fig.add_trace(go.Bar(x=split_df["dataset"], y=split_df["test_rows"], name="Test"), row=3, col=1)

    if not model_df.empty:
        fig.add_trace(go.Bar(x=model_df["model"], y=model_df["runtime"], name="Runtime (s)"), row=3, col=2)
        fig.add_trace(go.Scatter(x=model_df["model"], y=model_df["score"], mode="lines+markers", name="Score"), row=3, col=2)

    if not improve_df.empty:
        fig.add_trace(go.Bar(x=improve_df["model"], y=improve_df["delta"], name="Delta"), row=4, col=1)
        fig.add_trace(go.Scatter(x=improve_df["model"], y=improve_df["baseline"], mode="lines+markers", name="Baseline"), row=4, col=1)
        fig.add_trace(go.Scatter(x=improve_df["model"], y=improve_df["current"], mode="lines+markers", name="Current"), row=4, col=1)

    if not health_df.empty:
        fig.add_trace(go.Bar(x=health_df["metric"], y=health_df["value"], name="Health"), row=4, col=2)
    else:
        fig.add_trace(go.Bar(x=["pipeline"], y=[1], name="OK"), row=4, col=2)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1400,
        width=1800,
        barmode="group",
        legend_orientation="h",
        legend_y=-0.08
    )

    return fig


def controller_predict_update_dashboard(
    model_name,
    context,
    model_dir,
    incoming_data=None,
    new_raw_tables=None,
    update_model=False,
    return_proba=False,
    return_dashboard=False,
    run_metrics=None,
    dashboard_title="Pipeline Diagnostics Dashboard",
    
):
    """
    Controller for:
    - optionally updating the model with new raw tables
    - loading the saved model package
    - predicting on an already-prepared dataframe
    - optionally returning the dashboard

    Assumption:
    - incoming_data is already in the same feature shape the model expects,
      or at least compatible enough that the saved package's feature_names
      can be used to align it.
    """

    if update_model:
        if new_raw_tables is None:
            raise ValueError("new_raw_tables is required when update_model=True")

        context = ingest_new_data_and_retrain(
            new_raw_tables=new_raw_tables,
            context=context,
            save_dir=model_dir
        )

    package = context.get("trained_packages", {}).get(model_name)
    if package is None:
        package = import_model_package(model_dir, f"{model_name}.pkl")

    model = package["model"]

    if incoming_data is None:
        if return_dashboard:
            return build_one_page_pipeline_dashboard(
                context,
                run_metrics=run_metrics,
                title=dashboard_title
            )
        return package

    X = incoming_data.copy()

    expected_features = package.get("feature_names")
    if expected_features is not None:
        X = X.reindex(columns=expected_features, fill_value=0)

    if return_proba and hasattr(model, "predict_proba"):
        prediction_output = model.predict_proba(X)
    else:
        prediction_output = model.predict(X)

    if return_dashboard:
        dashboard = build_one_page_pipeline_dashboard(
            context,
            run_metrics=run_metrics,
            title=dashboard_title
        )
        return {
            "prediction": prediction_output,
            "dashboard": dashboard,
            "model_name": model_name,
        }

    return prediction_output