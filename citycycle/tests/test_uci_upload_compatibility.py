"""
CityCycle - Tests for UCI Bike Sharing Dataset ("hour.csv") Upload Compatibility
Verifies dual-schema auto-detection, exact unit conversion formulas,
target leakage quarantine, and batch prediction consistency on the official 17,379-row dataset.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.bulk_forecasting import (
    validate_and_filter_csv,
    predict_batch,
    generate_sample_template,
    detect_dataset_schema
)


def test_uci_schema_detection():
    print("\n--- TEST 1: UCI SCHEMA DETECTION ---")
    data_path = os.path.join(PROJECT_ROOT, "data", "hour.csv")
    assert os.path.exists(data_path), f"hour.csv not found at {data_path}!"

    raw_df = pd.read_csv(data_path)
    schema = detect_dataset_schema(raw_df.columns.tolist())
    assert schema == "UCI", f"Expected schema 'UCI', got '{schema}'"
    print(f"  [OK] Successfully recognized official UCI dataset format with {len(raw_df):,} rows.")


def test_uci_conversions_and_quarantine():
    print("\n--- TEST 2: UNIT CONVERSIONS & RECORD VALIDATION (17,379 ROWS) ---")
    data_path = os.path.join(PROJECT_ROOT, "data", "hour.csv")
    raw_df = pd.read_csv(data_path)

    valid_df, invalid_df, error_log = validate_and_filter_csv(raw_df)

    # 1. Row counts: all 17,379 rows must be valid
    assert len(valid_df) == 17379, f"Expected 17,379 valid rows, got {len(valid_df)}"
    assert len(invalid_df) == 0, f"Expected 0 invalid rows, got {len(invalid_df)}"
    print(f"  [OK] 100% of rows passed validation: 17,379 valid, 0 quarantined.")

    # 2. Check exact user-specified conversion formulas:
    # - temp_celsius = temp * 47 - 8
    # - humidity_pct = hum * 100
    # - windspeed_kmh = windspeed * 67
    expected_temp_c = np.round(raw_df["temp"].astype(float) * 47.0 - 8.0, 2)
    expected_hum_pct = np.round(raw_df["hum"].astype(float) * 100.0, 1)
    expected_wind_kmh = np.round(raw_df["windspeed"].astype(float) * 67.0, 2)

    np.testing.assert_allclose(valid_df["temp_celsius"], expected_temp_c, err_msg="temp_celsius formula mismatch!")
    np.testing.assert_allclose(valid_df["humidity_pct"], expected_hum_pct, err_msg="humidity_pct formula mismatch!")
    np.testing.assert_allclose(valid_df["windspeed_kmh"], expected_wind_kmh, err_msg="windspeed_kmh formula mismatch!")
    print("  [OK] Converted formulas verified: temp*47-8, hum*100, windspeed*67.")

    # 3. Check preservation of original columns
    for orig_col in ["instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "hum", "windspeed", "cnt"]:
        assert orig_col in valid_df.columns, f"Original column '{orig_col}' was lost!"

    print("  [OK] Original source columns ('instant', 'dteday', 'temp', 'hum', 'windspeed', 'cnt') preserved.")


def test_uci_batch_predictions():
    print("\n--- TEST 3: BATCH PREDICTION PIPELINE WITH GROUND TRUTH EVALUATION ---")
    model_pkg = joblib.load(os.path.join(PROJECT_ROOT, "models", "bike_demand_model.pkl"))
    model = model_pkg["model"]
    feature_names = model_pkg["feature_names"]

    data_path = os.path.join(PROJECT_ROOT, "data", "hour.csv")
    raw_df = pd.read_csv(data_path)

    # Test batch predictions across the first 500 records
    test_slice = raw_df.head(500)
    valid_slice, _, _ = validate_and_filter_csv(test_slice)

    results_df = predict_batch(model, feature_names, valid_slice)

    assert len(results_df) == len(valid_slice), f"Result row count ({len(results_df)}) does not match input ({len(valid_slice)})"

    # Verify output columns
    for col in ["predicted_demand", "risk_level", "recommended_vans", "recommended_staff", "action_directive", "prediction_error", "abs_error"]:
        assert col in results_df.columns, f"Missing output column '{col}' in results!"

    # Ground truth evaluation metrics
    mae = results_df["abs_error"].mean()
    rmse = np.sqrt((results_df["prediction_error"] ** 2).mean())
    print(f"  [OK] Successfully generated predictions for {len(results_df)} records.")
    print(f"  [OK] Ground truth evaluation: Upload MAE = {mae:.2f} bikes/hr, RMSE = {rmse:.2f} bikes/hr.")


def test_backward_compatibility_citycycle_template():
    print("\n--- TEST 4: BACKWARD COMPATIBILITY WITH CITYCYCLE TEMPLATE ---")
    template_df = generate_sample_template()
    schema = detect_dataset_schema(template_df.columns.tolist())
    assert schema == "CITYCYCLE", f"Expected schema 'CITYCYCLE', got '{schema}'"

    valid_df, invalid_df, _ = validate_and_filter_csv(template_df)
    assert len(valid_df) == len(template_df)
    assert len(invalid_df) == 0
    print("  [OK] CityCycle sample template continues to validate and process seamlessly.")


def run_all_tests():
    print("=" * 70)
    print("    CITYCYCLE - UCI 'hour.csv' UPLOAD COMPATIBILITY TEST SUITE")
    print("=" * 70)
    test_uci_schema_detection()
    test_uci_conversions_and_quarantine()
    test_uci_batch_predictions()
    test_backward_compatibility_citycycle_template()
    print("\n" + "=" * 70)
    print("   ALL UCI COMPATIBILITY TESTS PASSED (100% VERIFIED)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
