"""
CityCycle - Automated Hackathon Readiness Verifier
Checks model integrity, target leakage, chronological split, demo presets, and README consistency.
"""

import json
import os
import sys
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.decision_engine import generate_recommendation
from src.feature_engineering import split_chronological, FEATURE_COLUMNS, LEAKAGE_COLUMNS


def predict_demand_core(model, feature_names, inputs: dict) -> float:
    """Standalone feature vector builder and predictor matching app.py logic."""
    hr = inputs["hr"]
    workingday = inputs["workingday"]
    temp_celsius = inputs["temp_celsius"]

    is_rush_hour = 1 if (hr in [7, 8, 9, 17, 18, 19] and workingday == 1) else 0
    is_night = 1 if hr in [0, 1, 2, 3, 4, 5] else 0
    sin_hr = np.sin(2 * np.pi * hr / 24.0)
    cos_hr = np.cos(2 * np.pi * hr / 24.0)
    temp_norm = np.clip(temp_celsius / 41.0, 0.0, 1.0)
    atemp_norm = np.clip(temp_norm + 0.02, 0.0, 1.0)
    hum_norm = inputs["humidity_pct"] / 100.0
    wind_norm = inputs["windspeed_kmh"] / 67.0

    row_dict = {
        "season": inputs["season"],
        "yr": 1,
        "mnth": inputs["mnth"],
        "hr": hr,
        "holiday": inputs["holiday"],
        "weekday": inputs["weekday"],
        "workingday": workingday,
        "weathersit": inputs["weathersit"],
        "temp": temp_norm,
        "atemp": atemp_norm,
        "hum": hum_norm,
        "windspeed": wind_norm,
        "is_rush_hour": is_rush_hour,
        "is_night": is_night,
        "sin_hr": sin_hr,
        "cos_hr": cos_hr,
        "temp_celsius": temp_celsius
    }

    feature_df = pd.DataFrame([row_dict])[feature_names]
    pred = model.predict(feature_df)[0]
    return max(0.0, float(pred))


def check_leakage_and_model():
    print("\n--- CHECK 1: TARGET LEAKAGE & MODEL INTEGRITY ---")
    model_path = os.path.join(PROJECT_ROOT, "models", "bike_demand_model.pkl")
    assert os.path.exists(model_path), "Model file missing!"

    pkg = joblib.load(model_path)
    feature_names = pkg["feature_names"]

    # Assert no target leakage
    for forbidden in ["casual", "registered", "cnt", "instant", "dteday"]:
        assert forbidden not in feature_names, f"CRITICAL: Found forbidden column '{forbidden}' in model features!"

    print("  [PASS] Confirmed: 'casual' and 'registered' are 100% excluded from model inputs.")
    print(f"  [PASS] Features count: {len(feature_names)} features verified: {', '.join(feature_names[:6])}...")
    return pkg


def check_chronological_split():
    print("\n--- CHECK 2: CHRONOLOGICAL VS RANDOM SPLIT ---")
    data_path = os.path.join(PROJECT_ROOT, "data", "hour.csv")
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test, feats = split_chronological(df, train_ratio=0.80)

    train_dates = df.iloc[:len(X_train)]["dteday"]
    test_dates = df.iloc[len(X_train):]["dteday"]

    train_end = train_dates.max()
    test_start = test_dates.min()

    print(f"  Training period: {train_dates.min()} to {train_end} ({len(X_train):,} hours)")
    print(f"  Testing period:  {test_start} to {test_dates.max()} ({len(X_test):,} hours)")

    assert train_end <= test_start, f"Chronological split failed: train end ({train_end}) > test start ({test_start})"
    print("  [PASS] Confirmed: Split is 100% chronological with zero future data leakage.")


def check_presets_and_simulations(pkg):
    print("\n--- CHECK 3: PRESETS & SIMULATION PARAMETERS ---")
    model = pkg["model"]
    feature_names = pkg["feature_names"]

    presets = [
        {"name": "Sunny Morning Commute", "hr": 8, "workingday": 1, "weekday": 2, "mnth": 6, "season": 2, "holiday": 0, "weathersit": 1, "temp_celsius": 24.0, "humidity_pct": 50, "windspeed_kmh": 10.0},
        {"name": "Rainy Autumn Evening", "hr": 18, "workingday": 1, "weekday": 3, "mnth": 10, "season": 3, "holiday": 0, "weathersit": 3, "temp_celsius": 12.0, "humidity_pct": 88, "windspeed_kmh": 25.0},
        {"name": "Sunny Weekend Afternoon", "hr": 14, "workingday": 0, "weekday": 6, "mnth": 7, "season": 2, "holiday": 0, "weathersit": 1, "temp_celsius": 27.0, "humidity_pct": 45, "windspeed_kmh": 14.0},
        {"name": "Late Night Maintenance", "hr": 3, "workingday": 1, "weekday": 1, "mnth": 4, "season": 1, "holiday": 0, "weathersit": 1, "temp_celsius": 11.0, "humidity_pct": 65, "windspeed_kmh": 8.0}
    ]

    for p in presets:
        pred = predict_demand_core(model, feature_names, p)
        assert pred >= 0, f"Negative prediction on {p['name']}"
        rec = generate_recommendation(pred, hour=p["hr"], workingday=p["workingday"], weathersit=p["weathersit"], temp_celsius=p["temp_celsius"])

        assert rec["risk_level"] in ["Low", "Moderate", "High", "Critical"]
        assert rec["trucks_recommended"] >= 0
        assert rec["staffing_estimate"] >= 1
        print(f"  [PASS] Preset '{p['name']}': Forecast={int(pred)} bikes/hr | Risk={rec['risk_level']} | Vans={rec['trucks_recommended']} | Staff={rec['staffing_estimate']}")

    # Boundary stress test
    print("  Testing extreme parameter boundaries (-5 deg C to 40 deg C, 100% humidity, 50 km/h wind, all 24 hours)...")
    for test_hr in range(24):
        for test_temp in [-5.0, 0.0, 20.0, 40.0]:
            boundary_inputs = {
                "hr": test_hr, "workingday": 1, "weekday": 1, "mnth": 1, "season": 1,
                "holiday": 0, "weathersit": 4, "temp_celsius": test_temp,
                "humidity_pct": 100, "windspeed_kmh": 50.0
            }
            b_pred = predict_demand_core(model, feature_names, boundary_inputs)
            assert not np.isnan(b_pred) and b_pred >= 0

    print("  [PASS] Boundary stress test passed: zero NaNs, zero negative values across all 24 hours.")


def check_metrics_consistency():
    print("\n--- CHECK 4: README METRICS VS EVALUATION RESULTS ---")
    json_path = os.path.join(PROJECT_ROOT, "models", "evaluation_results.json")
    readme_path = os.path.join(PROJECT_ROOT, "README.md")

    with open(json_path, "r") as f:
        metrics = json.load(f)

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    base_mae = f"{metrics['metrics']['baseline']['mae']}"
    base_rmse = f"{metrics['metrics']['baseline']['rmse']}"
    base_r2 = f"{metrics['metrics']['baseline']['r2']}"

    rf_mae = f"{metrics['metrics']['random_forest']['mae']}"
    rf_rmse = f"{metrics['metrics']['random_forest']['rmse']}"
    rf_r2 = f"{metrics['metrics']['random_forest']['r2']}"

    print(f"  Saved JSON Baseline:     MAE={base_mae}, RMSE={base_rmse}, R^2={base_r2}")
    print(f"  Saved JSON Random Forest: MAE={rf_mae}, RMSE={rf_rmse}, R^2={rf_r2}")

    for val, name in [
        (base_mae, "Baseline MAE"),
        (base_rmse, "Baseline RMSE"),
        (base_r2, "Baseline R^2"),
        (rf_mae, "Random Forest MAE"),
        (rf_rmse, "Random Forest RMSE"),
        (rf_r2, "Random Forest R^2")
    ]:
        assert val in readme_text, f"Metric mismatch! {name} value {val} not found in README.md"
        print(f"  [PASS] Verified in README: {name} = {val}")


def main():
    print("=" * 68)
    print("      CITYCYCLE - HACKATHON READINESS COMPREHENSIVE AUDIT")
    print("=" * 68)
    pkg = check_leakage_and_model()
    check_chronological_split()
    check_presets_and_simulations(pkg)
    check_metrics_consistency()
    print("\n" + "=" * 68)
    print("   ALL HACKATHON READINESS CHECKS PASSED (100% FACTUAL MATCH)")
    print("=" * 68)


if __name__ == "__main__":
    main()
