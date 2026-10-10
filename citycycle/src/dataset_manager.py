"""
CityCycle - Dataset Manager & Client Model Engine
Manages dataset cataloging, switching, metadata profiling, model compatibility checking,
stale cache prevention, and isolated client-specific model training.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.bulk_forecasting import (
    validate_and_filter_csv,
    predict_batch,
    detect_dataset_schema
)
from src.feature_engineering import FEATURE_COLUMNS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_default_uci_dataset() -> pd.DataFrame:
    """Loads the default official UCI benchmark dataset."""
    hour_path = os.path.join(DATA_DIR, "hour.csv")
    bike_path = os.path.join(DATA_DIR, "bike_sharing_hourly.csv")
    if os.path.exists(hour_path):
        return pd.read_csv(hour_path)
    elif os.path.exists(bike_path):
        return pd.read_csv(bike_path)
    raise FileNotFoundError("Default dataset (hour.csv) not found in data/ directory!")


def check_model_compatibility(valid_df: pd.DataFrame, required_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Checks whether valid_df can produce the exact features expected by the pretrained model.
    Returns: (is_compatible, missing_or_problematic_features)
    """
    if valid_df.empty:
        return False, ["Dataset is empty."]

    missing = []
    # Check if necessary raw components exist to build the 17 features
    if "hr" not in valid_df.columns:
        missing.append("Hour feature ('hr')")
    if "workingday" not in valid_df.columns:
        missing.append("Working day flag ('workingday')")
    if "weathersit" not in valid_df.columns:
        missing.append("Weather condition ('weathersit')")
    if "season" not in valid_df.columns:
        missing.append("Season indicator ('season')")
    if "mnth" not in valid_df.columns:
        missing.append("Month indicator ('mnth')")
    if "weekday" not in valid_df.columns:
        missing.append("Day of week ('weekday')")

    # Weather variables: needs either UCI normalized (temp, hum, windspeed) or converted
    has_uci_weather = all(col in valid_df.columns for col in ["temp", "hum", "windspeed"])
    has_converted_weather = all(col in valid_df.columns for col in ["temp_celsius", "humidity_pct", "windspeed_kmh"])

    if not (has_uci_weather or has_converted_weather):
        missing.append("Weather metrics (requires either normalized 'temp, hum, windspeed' or 'temp_celsius, humidity_pct, windspeed_kmh')")

    is_compatible = len(missing) == 0
    return is_compatible, missing


def profile_dataset(name: str, df: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
    """Profiles a dataset and checks validation and model compatibility."""
    valid_df, invalid_df, error_log = validate_and_filter_csv(df)
    schema = detect_dataset_schema(df.columns.tolist())
    is_compatible, missing_features = check_model_compatibility(valid_df, FEATURE_COLUMNS)

    # Detect possible target column
    target_col = None
    for candidate in ["cnt", "demand", "rentals", "trips", "riders", "count"]:
        if candidate in df.columns:
            target_col = candidate
            break

    return {
        "id": dataset_id,
        "name": name,
        "raw_df": df,
        "valid_df": valid_df,
        "invalid_df": invalid_df,
        "validation_messages": error_log,
        "schema": schema,
        "target_col": target_col,
        "is_compatible": is_compatible,
        "missing_features": missing_features,
        "total_rows": len(df),
        "valid_rows": len(valid_df),
        "invalid_rows": len(invalid_df),
        "columns": df.columns.tolist(),
        "has_ground_truth": target_col is not None
    }


def train_client_model(
    valid_df: pd.DataFrame,
    target_col: str,
    model_type: str = "RandomForest"
) -> Dict[str, Any]:
    """
    Trains a client-specific regression model using chronological 80/20 train/test splitting.
    Strictly excludes the target and leakage columns from prediction inputs.
    """
    if valid_df.empty or target_col not in valid_df.columns:
        raise ValueError(f"Cannot train model: valid dataset is empty or target column '{target_col}' is missing.")

    # Sort chronologically if date/time present
    df = valid_df.copy()
    if "dteday" in df.columns and "hr" in df.columns:
        df = df.sort_values(by=["dteday", "hr"]).reset_index(drop=True)
    elif "instant" in df.columns:
        df = df.sort_values(by="instant").reset_index(drop=True)

    # Feature selection: use standard available predictors, strictly excluding leakage columns
    forbidden = ["casual", "registered", target_col, "cnt", "instant", "dteday", "record_id"]
    candidate_features = [col for col in df.columns if col not in forbidden and np.issubdtype(df[col].dtype, np.number)]

    if len(candidate_features) < 3:
        raise ValueError(f"Insufficient numeric features for training. Found: {candidate_features}")

    # Chronological Split (80% train, 20% test)
    n = len(df)
    n_train = max(10, int(n * 0.80))
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    X_train = train_df[candidate_features]
    y_train = train_df[target_col].astype(float)
    X_test = test_df[candidate_features]
    y_test = test_df[target_col].astype(float)

    # Train Model
    if model_type == "Ridge":
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = np.maximum(0, model.predict(X_test))

    # Evaluate against baseline
    baseline_pred = np.full_like(y_test, y_train.mean())
    mae_base = mean_absolute_error(y_test, baseline_pred)
    mae_model = mean_absolute_error(y_test, preds)
    rmse_model = np.sqrt(mean_squared_error(y_test, preds))
    r2_model = r2_score(y_test, preds)

    return {
        "model": model,
        "model_type": model_type,
        "feature_names": candidate_features,
        "target_col": target_col,
        "metrics": {
            "mae": round(mae_model, 2),
            "rmse": round(rmse_model, 2),
            "r2": round(r2_model, 4),
            "baseline_mae": round(mae_base, 2)
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }


class DatasetManager:
    """Manages active dataset selection, caching, and client models."""

    def __init__(self):
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.client_models: Dict[str, Dict[str, Any]] = {}
        self.active_id: str = "default_uci"
        self._init_default()

    def _init_default(self):
        try:
            default_df = load_default_uci_dataset()
            profile = profile_dataset("🚲 Default UCI Benchmark (Capital Bikeshare)", default_df, "default_uci")
            self.catalog["default_uci"] = profile
        except Exception as e:
            print(f"Warning: could not load default dataset: {e}")

    def register_client_dataset(self, name: str, df: pd.DataFrame, dataset_id: Optional[str] = None) -> str:
        """Profiles and adds a new client dataset into the catalog."""
        if dataset_id is None:
            dataset_id = f"client_{len(self.catalog)}_{abs(hash(name)) % 10000}"
        profile = profile_dataset(name, df, dataset_id)
        self.catalog[dataset_id] = profile
        self.active_id = dataset_id
        return dataset_id

    def set_active_dataset(self, dataset_id: str):
        """Switches the active dataset."""
        if dataset_id in self.catalog:
            self.active_id = dataset_id
        else:
            raise KeyError(f"Dataset ID '{dataset_id}' not found in catalog.")

    def get_active_profile(self) -> Dict[str, Any]:
        """Returns the profile of the currently active dataset."""
        if self.active_id not in self.catalog:
            self.active_id = "default_uci"
        return self.catalog[self.active_id]

    def get_dataset_options(self) -> Dict[str, str]:
        """Returns a mapping of display names to dataset IDs."""
        return {info["name"]: info["id"] for info in self.catalog.values()}
