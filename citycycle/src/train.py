"""
CityCycle - Model Training and Evaluation Module
Trains a baseline and Random Forest regressor on chronological splits and evaluates test performance.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_engineering import split_chronological, FEATURE_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_dataset_path() -> str:
    """Locates the dataset file (supporting hour.csv or bike_sharing_hourly.csv)."""
    hour_path = os.path.join(DATA_DIR, "hour.csv")
    bike_path = os.path.join(DATA_DIR, "bike_sharing_hourly.csv")
    if os.path.exists(hour_path):
        return hour_path
    elif os.path.exists(bike_path):
        return bike_path
    raise FileNotFoundError(f"Neither hour.csv nor bike_sharing_hourly.csv found in {DATA_DIR}")


def train_and_evaluate():
    data_path = get_dataset_path()
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    print(f"Dataset successfully loaded: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns.")

    # Chronological Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test, feature_names = split_chronological(raw_df, train_ratio=0.80)

    # Verification: Target leakage check
    leakage_detected = [col for col in ["casual", "registered", "cnt"] if col in X_train.columns]
    if leakage_detected:
        raise ValueError(f"CRITICAL TARGET LEAKAGE: Found {leakage_detected} in training features!")
    print(f"Leakage Check: PASSED (casual and registered are completely excluded).")
    print(f"Training set:   {X_train.shape[0]:,} samples")
    print(f"Test set:       {X_test.shape[0]:,} samples (latest chronological period)")
    print(f"Features used ({len(feature_names)}): {', '.join(feature_names)}")

    # 1. Baseline Model (Ridge Regression)
    print("\nTraining Baseline Model (Linear Ridge Regression)...")
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    y_pred_baseline = np.maximum(0, baseline.predict(X_test))

    mae_base = mean_absolute_error(y_test, y_pred_baseline)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    r2_base = r2_score(y_test, y_pred_baseline)

    # 2. Random Forest Regressor
    print("Training Optimized Random Forest Regressor (100 trees)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=22,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = np.maximum(0, rf_model.predict(X_test))

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    # Calculate Improvement
    mae_improvement = ((mae_base - mae_rf) / mae_base) * 100
    rmse_improvement = ((rmse_base - rmse_rf) / rmse_base) * 100

    # Print Results Table
    print("\n" + "=" * 68)
    print("           CITYCYCLE - MODEL EVALUATION RESULTS (TEST PERIOD)")
    print("=" * 68)
    print(f"{'Metric':<15} | {'Baseline (Linear)':<20} | {'Random Forest':<20}")
    print("-" * 68)
    print(f"{'MAE (bikes)':<15} | {mae_base:<20.2f} | {mae_rf:<20.2f}")
    print(f"{'RMSE (bikes)':<15} | {rmse_base:<20.2f} | {rmse_rf:<20.2f}")
    print(f"{'R-squared (R²)':<15} | {r2_base:<20.4f} | {r2_rf:<20.4f}")
    print("-" * 68)
    print(f"Random Forest MAE Error Reduction:  {mae_improvement:.1f}%")
    print(f"Random Forest RMSE Error Reduction: {rmse_improvement:.1f}%")
    print("=" * 68)

    # Feature Importance Top 5
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 5 Most Important Features in Random Forest:")
    for rank, (feat, imp) in enumerate(importances.head(5).items(), 1):
        print(f"  {rank}. {feat:<15}: {imp * 100:.2f}%")

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_payload = {
        "model": rf_model,
        "feature_names": feature_names,
        "target_name": TARGET_COLUMN,
        "metrics": {
            "baseline": {"mae": mae_base, "rmse": rmse_base, "r2": r2_base},
            "random_forest": {"mae": mae_rf, "rmse": rmse_rf, "r2": r2_rf}
        }
    }
    model_path = os.path.join(MODELS_DIR, "bike_demand_model.pkl")
    joblib.dump(model_payload, model_path, compress=3)
    print(f"\nTrained model package saved with compression to: {model_path}")

    results_json = {
        "dataset_rows": len(raw_df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_names,
        "metrics": {
            "baseline": {
                "mae": round(mae_base, 2),
                "rmse": round(rmse_base, 2),
                "r2": round(r2_base, 4)
            },
            "random_forest": {
                "mae": round(mae_rf, 2),
                "rmse": round(rmse_rf, 2),
                "r2": round(r2_rf, 4)
            }
        },
        "top_features": {k: round(v, 4) for k, v in importances.items()}
    }
    results_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Evaluation metrics JSON saved to: {results_path}")

    return results_json


if __name__ == "__main__":
    train_and_evaluate()
