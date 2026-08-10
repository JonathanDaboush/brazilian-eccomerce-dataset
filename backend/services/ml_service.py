from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from services.upload_service import get_dataset_file_path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_VERSIONS_DIR = BASE_DIR / "model_versions"

MODEL_FILES = {
    "delivery_delay": "delivery_delay.pkl",
    "order_cancellation": "order_cancellation.pkl",
    "review_prediction": "review_prediction.pkl",
    "demand_forecasting": "demand_forecasting.pkl",
    "product_recommendation": "product_recommendation.pkl",
    "customer_purchase_prediction": "customer_purchase_prediction.pkl",
}

TARGET_COLS = {
    "delivery_delay": "late_delivery",
    "order_cancellation": "cancelled",
    "review_prediction": "review_score",
    "demand_forecasting": "units_sold",
    "product_recommendation": None,
    "customer_purchase_prediction": "future_purchase",
}

TIME_PRIORITY = ("order_purchase_timestamp", "review_creation_date", "purchase_date", "event_timestamp", "month")
CLASSIFICATION_MODELS = {"delivery_delay", "order_cancellation", "customer_purchase_prediction"}
REGRESSION_MODELS = {"review_prediction", "demand_forecasting"}


@lru_cache(maxsize=16)
def _load_model(name: str):
    path = MODELS_DIR / MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _clear_model_cache(name: str | None = None) -> None:
    _load_model.cache_clear()
    if name is None:
        _training_frame.cache_clear()
    else:
        _training_frame.cache_clear()


@lru_cache(maxsize=16)
def _training_frame(name: str) -> pd.DataFrame:
    path = get_dataset_file_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    return pd.read_csv(path)


def _load_package_if_available(name: str) -> dict[str, Any] | None:
    try:
        obj = _load_model(name)
    except FileNotFoundError:
        return None
    return obj if isinstance(obj, dict) else {"model": obj}


def _feature_schema(model_name: str) -> list[dict]:
    try:
        df = _training_frame(model_name)
    except FileNotFoundError:
        return []
    target = TARGET_COLS.get(model_name)
    schema = []
    for col in df.columns:
        if col == target:
            continue
        dtype = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        sample_values: list[str] = []
        if not is_numeric:
            sample_values = [str(v) for v in df[col].dropna().astype(str).unique()[:5].tolist()]
        schema.append(
            {
                "name": col,
                "dtype": dtype,
                "numeric": is_numeric,
                "sample_values": sample_values,
                "default": (
                    float(df[col].median()) if is_numeric and not df[col].dropna().empty
                    else (df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else None)
                ),
            }
        )
    return schema


def available_models() -> list[dict[str, Any]]:
    out = []
    for name, filename in MODEL_FILES.items():
        package = _load_package_if_available(name)
        metadata = package or {}
        out.append(
            {
                "name": name,
                "model_file": filename,
                "available": (MODELS_DIR / filename).exists(),
                "training_data_available": get_dataset_file_path(name).exists(),
                "feature_schema": _feature_schema(name),
                "target_column": TARGET_COLS.get(name),
                "trained_at": metadata.get("trained_at"),
                "training_dataset_path": metadata.get("training_dataset_path"),
                "metrics": metadata.get("metrics"),
                "model_version_path": metadata.get("model_version_path"),
            }
        )
    return out


def _build_feature_frame(model_name: str, features: dict | None) -> pd.DataFrame:
    df = _training_frame(model_name)
    target = TARGET_COLS[model_name]
    feature_cols = [c for c in df.columns if c != target]

    features = features or {}
    row = {}
    for col in feature_cols:
        if col in features:
            row[col] = features[col]
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(df[col].median()) if not df[col].dropna().empty else 0.0
        else:
            mode = df[col].mode(dropna=True)
            row[col] = mode.iloc[0] if not mode.empty else "unknown"

    frame = pd.DataFrame([row], columns=feature_cols)
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = float(df[col].median()) if not df[col].dropna().empty else 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(fill)
    return frame


def _apply_package_preprocessing(package: dict, frame: pd.DataFrame) -> pd.DataFrame:
    transformed = frame.copy()

    for col in package.get("dropped_columns", []) or []:
        if col in transformed.columns:
            transformed = transformed.drop(columns=[col])

    fill_values = package.get("fill_values", {}) or {}
    for col, val in fill_values.items():
        if col in transformed.columns:
            transformed[col] = transformed[col].fillna(val)

    clipping_values = package.get("clipping_values", {}) or {}
    for col, bounds in clipping_values.items():
        if col in transformed.columns and isinstance(bounds, dict):
            transformed[col] = pd.to_numeric(transformed[col], errors="coerce")
            transformed[col] = transformed[col].clip(lower=bounds.get("low"), upper=bounds.get("high"))

    encoder = package.get("encoder")
    if encoder is not None and hasattr(encoder, "feature_names_in_"):
        enc_cols = list(encoder.feature_names_in_)
        for col in enc_cols:
            if col not in transformed.columns:
                transformed[col] = "unknown"
        transformed[enc_cols] = encoder.transform(transformed[enc_cols])

    scaler = package.get("scaler")
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        scale_cols = list(scaler.feature_names_in_)
        for col in scale_cols:
            if col not in transformed.columns:
                transformed[col] = 0.0
        transformed[scale_cols] = scaler.transform(transformed[scale_cols])

    return transformed


def predict(model_name: str, features: dict | None = None) -> dict[str, Any]:
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'")

    model_obj = _load_model(model_name)
    package = model_obj if isinstance(model_obj, dict) else {"model": model_obj}
    model = package["model"]
    frame = _build_feature_frame(model_name, features)

    if model_name == "product_recommendation":
        transformed = _apply_package_preprocessing(package, frame)
        feature_names = package.get("feature_names")
        if feature_names:
            transformed = transformed.reindex(columns=feature_names, fill_value=0)
        distances, indices = model.kneighbors(transformed)
        return {
            "model": model_name,
            "prediction_type": "nearest_neighbors",
            "neighbor_indices": indices.tolist()[0],
            "neighbor_distances": distances.tolist()[0],
            "trained_at": package.get("trained_at"),
        }

    transformed = _apply_package_preprocessing(package, frame)
    feature_names = package.get("feature_names")
    if feature_names:
        transformed = transformed.reindex(columns=feature_names, fill_value=0)

    y_pred = model.predict(transformed)
    result: dict[str, Any] = {
        "model": model_name,
        "prediction": float(y_pred[0]) if np.isscalar(y_pred[0]) else y_pred[0],
        "used_defaults_for_missing_features": True,
        "feature_columns_count": int(transformed.shape[1]),
        "trained_at": package.get("trained_at"),
        "metrics": package.get("metrics"),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(transformed)
        if getattr(proba, "shape", [0, 0])[1] == 2:
            result["probability_positive"] = float(proba[0][1])
        else:
            result["probabilities"] = proba[0].tolist()

    return result


def _pick_time_column(df: pd.DataFrame, feature_cols: list[str]) -> str | None:
    for candidate in TIME_PRIORITY:
        if candidate in feature_cols:
            return candidate
    for col in feature_cols:
        lowered = col.lower()
        if "date" in lowered or "time" in lowered or "month" == lowered:
            return col
    return None


def _prepare_xy(model_name: str) -> tuple[pd.DataFrame, pd.Series, str | None]:
    df = _training_frame(model_name).copy()
    target = TARGET_COLS[model_name]
    if target is None:
        raise ValueError(f"{model_name} does not support supervised retraining")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing from {model_name} dataset")

    feature_cols = [c for c in df.columns if c != target]
    time_col = _pick_time_column(df, feature_cols)
    if time_col is not None:
        converted = pd.to_datetime(df[time_col], errors="coerce")
        if converted.notna().any():
            df = df.assign(**{time_col: converted}).sort_values(time_col).reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y, time_col


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _default_estimator(model_name: str):
    if model_name in CLASSIFICATION_MODELS:
        return LogisticRegression(max_iter=2000)
    if model_name in REGRESSION_MODELS:
        return RandomForestRegressor(n_estimators=200, random_state=42)
    raise ValueError(f"No default estimator configured for {model_name}")


def _select_estimator(model_name: str):
    package = _load_package_if_available(model_name)
    current_model = package.get("model") if package else None
    if current_model is not None and not isinstance(current_model, Pipeline):
        try:
            return clone(current_model)
        except Exception:
            pass
    return _default_estimator(model_name)


def _split_dataset(model_name: str, X: pd.DataFrame, y: pd.Series, time_col: str | None):
    if len(X) < 5:
        raise ValueError(f"{model_name} dataset is too small to retrain safely")

    if time_col is not None and time_col in X.columns:
        split_idx = max(int(len(X) * 0.8), 1)
        split_idx = min(split_idx, len(X) - 1)
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:], "chronological"

    stratify = y if model_name in CLASSIFICATION_MODELS and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test, "random"


def retrain_model(model_name: str) -> dict[str, Any]:
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'")
    if model_name == "product_recommendation":
        raise ValueError("product_recommendation retraining is not wired through this API")

    X, y, time_col = _prepare_xy(model_name)
    X_train, X_test, y_train, y_test, split_strategy = _split_dataset(model_name, X, y, time_col)
    estimator = _select_estimator(model_name)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(X_train)),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)

    metrics: dict[str, Any] = {
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "split_strategy": split_strategy,
    }
    if model_name in CLASSIFICATION_MODELS:
        pred = pipeline.predict(X_test)
        metrics["accuracy"] = float(accuracy_score(y_test, pred))
        if y_test.nunique(dropna=True) > 1 and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    else:
        pred = pipeline.predict(X_test)
        metrics["r2"] = float(r2_score(y_test, pred))
        metrics["rmse"] = float(mean_squared_error(y_test, pred, squared=False))

    trained_at = datetime.now(timezone.utc).isoformat()
    version_id = f"{trained_at[:19].replace(':', '').replace('-', '')}-{model_name}"
    model_version_dir = MODEL_VERSIONS_DIR / model_name
    model_version_dir.mkdir(parents=True, exist_ok=True)
    version_path = model_version_dir / f"{version_id}.pkl"
    active_path = MODELS_DIR / MODEL_FILES[model_name]

    package = {
        "model": pipeline,
        "feature_names": list(X.columns),
        "trained_at": trained_at,
        "metrics": metrics,
        "target_column": TARGET_COLS[model_name],
        "training_dataset_path": str(get_dataset_file_path(model_name)),
        "model_version_path": str(version_path),
        "time_column_used": time_col,
    }

    with open(version_path, "wb") as fp:
        pickle.dump(package, fp)
    with open(active_path, "wb") as fp:
        pickle.dump(package, fp)

    _clear_model_cache(model_name)
    return {
        "ok": True,
        "model": model_name,
        "trained_at": trained_at,
        "metrics": metrics,
        "model_version_path": str(version_path),
        "active_model_path": str(active_path),
        "training_dataset_path": str(get_dataset_file_path(model_name)),
    }


def retrain_all_models() -> dict[str, Any]:
    runs = []
    for name in MODEL_FILES:
        if name == "product_recommendation":
            continue
        try:
            if get_dataset_file_path(name).exists():
                runs.append(retrain_model(name))
        except Exception as exc:
            runs.append({"ok": False, "model": name, "error": str(exc)})
    return {"runs": runs}


def recent_training_runs(limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not MODEL_VERSIONS_DIR.exists():
        return rows
    for model_dir in MODEL_VERSIONS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for artifact in sorted(model_dir.glob("*.pkl"), reverse=True):
            try:
                with open(artifact, "rb") as fp:
                    package = pickle.load(fp)
                rows.append(
                    {
                        "model": model_dir.name,
                        "artifact_path": str(artifact),
                        "trained_at": package.get("trained_at"),
                        "metrics": package.get("metrics"),
                        "training_dataset_path": package.get("training_dataset_path"),
                    }
                )
            except Exception:
                continue
    rows.sort(key=lambda item: item.get("trained_at") or "", reverse=True)
    return rows[:limit]
