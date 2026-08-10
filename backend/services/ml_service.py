from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TRAINING_DIR = BASE_DIR / "ml_training_data"

MODEL_FILES = {
    "delivery_delay": "delivery_delay.pkl",
    "order_cancellation": "order_cancellation.pkl",
    "review_prediction": "review_prediction.pkl",
    "demand_forecasting": "demand_forecasting.pkl",
    "product_recommendation": "product_recommendation.pkl",
}

TARGET_COLS = {
    "delivery_delay": "late_delivery",
    "order_cancellation": "cancelled",
    "review_prediction": "review_score",
    "demand_forecasting": "units_sold",
    "product_recommendation": None,
}


@lru_cache(maxsize=16)
def _load_model(name: str):
    path = MODELS_DIR / MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fp:
        return pickle.load(fp)


@lru_cache(maxsize=16)
def _training_frame(name: str) -> pd.DataFrame:
    path = TRAINING_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    return pd.read_csv(path)


def available_models() -> list[dict[str, str | bool]]:
    out = []
    for name, filename in MODEL_FILES.items():
        out.append(
            {
                "name": name,
                "model_file": filename,
                "available": (MODELS_DIR / filename).exists(),
                "training_data_available": (TRAINING_DIR / f"{name}.csv").exists(),
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
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(float(df[col].median()) if not df[col].dropna().empty else 0.0)
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
            low = bounds.get("low")
            high = bounds.get("high")
            transformed[col] = pd.to_numeric(transformed[col], errors="coerce")
            transformed[col] = transformed[col].clip(lower=low, upper=high)

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


def predict(model_name: str, features: dict | None = None) -> dict:
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'")

    model_obj = _load_model(model_name)
    package = model_obj if isinstance(model_obj, dict) else {"model": model_obj}
    model = package["model"]
    frame = _build_feature_frame(model_name, features)
    frame = _apply_package_preprocessing(package, frame)
    feature_names = package.get("feature_names")
    if feature_names:
        frame = frame.reindex(columns=feature_names, fill_value=0)

    if model_name == "product_recommendation":
        distances, indices = model.kneighbors(frame)
        return {
            "model": model_name,
            "prediction_type": "nearest_neighbors",
            "neighbor_indices": indices.tolist()[0],
            "neighbor_distances": distances.tolist()[0],
        }

    y_pred = model.predict(frame)
    result = {"model": model_name, "prediction": float(y_pred[0]) if np.isscalar(y_pred[0]) else y_pred[0]}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        if proba.shape[1] == 2:
            result["probability_positive"] = float(proba[0][1])
        else:
            result["probabilities"] = proba[0].tolist()

    result["used_defaults_for_missing_features"] = True
    result["feature_columns_count"] = int(frame.shape[1])
    return result
