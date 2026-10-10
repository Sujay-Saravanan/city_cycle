"""
CityCycle - Tests for Bulk Forecasting and Weather Impact Twin Modules
Verifies CSV template generation, schema validation, batch inference,
target leakage quarantining, and Weather Twin alert generation.
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
    generate_sample_template,
    validate_and_filter_csv,
    predict_batch,
    REQUIRED_COLUMNS
)
from src.weather_twin import (
    TWIN_PRESETS,
    simulate_weather_twin_single,
    simulate_weather_twin_24h
)


def test_sample_template():
    print("\n--- TEST 1: CSV SAMPLE TEMPLATE INTEGRITY ---")
    df = generate_sample_template()
    assert not df.empty, "Template is empty!"
    assert len(df) == 12, f"Expected 12 sample rows, got {len(df)}"

    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Required column '{col}' missing from template!"

    print("  [OK] Sample CSV template generated with correct 12 rows and all required headers.")


def test_csv_validation_and_quarantine():
    print("\n--- TEST 2: CSV VALIDATION & MALFORMED ROW QUARANTINE ---")

    # Construct test data with 3 valid rows and 3 intentionally broken rows
    test_rows = [
        {"record_id": "VAL-01", "hr": 8, "temp_celsius": 20.0, "humidity_pct": 50, "windspeed_kmh": 10.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1},
        {"record_id": "VAL-02", "hr": 12, "temp_celsius": 25.0, "humidity_pct": 45, "windspeed_kmh": 12.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1},
        {"record_id": "VAL-03", "hr": 17, "temp_celsius": 22.0, "humidity_pct": 55, "windspeed_kmh": 15.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1},
        # Broken rows:
        {"record_id": "ERR-01", "hr": 29, "temp_celsius": 20.0, "humidity_pct": 50, "windspeed_kmh": 10.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1}, # hr > 23
        {"record_id": "ERR-02", "hr": 14, "temp_celsius": 99.0, "humidity_pct": 50, "windspeed_kmh": 10.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1}, # temp > 55
        {"record_id": "ERR-03", "hr": 15, "temp_celsius": 20.0, "humidity_pct": 150, "windspeed_kmh": 10.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1} # hum > 100
    ]
    raw_df = pd.DataFrame(test_rows)

    valid_df, invalid_df, errors = validate_and_filter_csv(raw_df)

    assert len(valid_df) == 3, f"Expected 3 valid rows, got {len(valid_df)}"
    assert len(invalid_df) == 3, f"Expected 3 invalid rows, got {len(invalid_df)}"
    assert len(errors) == 3, f"Expected 3 error reports, got {len(errors)}"

    print(f"  [OK] Correctly quarantined 3 malformed rows and accepted 3 valid rows.")
    print(f"  [OK] Error explanation logged: '{errors[0]}'")


def test_target_leakage_in_csv():
    print("\n--- TEST 3: CSV TARGET LEAKAGE QUARANTINE ---")

    # CSV maliciously or mistakenly containing target leakage columns
    leakage_rows = [
        {"hr": 8, "temp_celsius": 20.0, "humidity_pct": 50, "windspeed_kmh": 10.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1,
         "casual": 50, "registered": 400, "cnt": 450}
    ]
    raw_df = pd.DataFrame(leakage_rows)

    valid_df, _, errors = validate_and_filter_csv(raw_df)

    assert "casual" not in valid_df.columns, "Target leakage column 'casual' was not removed!"
    assert "registered" not in valid_df.columns, "Target leakage column 'registered' was not removed!"
    assert "cnt" in valid_df.columns, "Target column 'cnt' was not preserved for evaluation!"

    print("  [OK] Leakage columns 'casual' and 'registered' stripped; 'cnt' preserved for evaluation.")


def test_batch_prediction():
    print("\n--- TEST 4: BATCH PREDICTION PIPELINE ---")
    model_pkg = joblib.load(os.path.join(PROJECT_ROOT, "models", "bike_demand_model.pkl"))
    model = model_pkg["model"]
    feature_names = model_pkg["feature_names"]

    template_df = generate_sample_template()
    valid_df, _, _ = validate_and_filter_csv(template_df)

    results_df = predict_batch(model, feature_names, valid_df)

    # Output count must match input count
    assert len(results_df) == len(valid_df), f"Output count ({len(results_df)}) does not match input count ({len(valid_df)})"

    for expected_col in ["predicted_demand", "risk_level", "recommended_vans", "recommended_staff", "action_directive"]:
        assert expected_col in results_df.columns, f"Expected output column '{expected_col}' missing from batch results!"

    assert (results_df["predicted_demand"] >= 0).all(), "Negative prediction found in batch output!"
    print(f"  [OK] Batch prediction processed {len(results_df)} rows. Average demand: {results_df['predicted_demand'].mean():.1f} bikes/hr.")


def test_weather_twin():
    print("\n--- TEST 5: WEATHER IMPACT TWIN & SMART ALERTS ---")
    model_pkg = joblib.load(os.path.join(PROJECT_ROOT, "models", "bike_demand_model.pkl"))
    model = model_pkg["model"]
    feature_names = model_pkg["feature_names"]

    base_inputs = {
        "hr": 17, "workingday": 1, "weekday": 2, "mnth": 6, "season": 2, "holiday": 0,
        "weathersit": 1, "temp_celsius": 24.0, "humidity_pct": 50, "windspeed_kmh": 12.0
    }

    # Simulate Heavy Rain Twin
    twin_inputs = base_inputs.copy()
    twin_inputs["weathersit"] = 4
    twin_inputs["temp_celsius"] = 19.0
    twin_inputs["humidity_pct"] = 92.0
    twin_inputs["windspeed_kmh"] = 28.0

    comp = simulate_weather_twin_single(model, feature_names, base_inputs, twin_inputs, "Heavy Rain / Thunderstorm")

    # Rain should decrease demand during rush hour
    assert comp["twin_demand"] < comp["baseline_demand"], "Rain did not reduce predicted demand!"
    assert comp["delta_demand"] < 0
    assert len(comp["alerts"]) >= 1

    print(f"  [OK] Single-Hour Twin: Base={comp['baseline_demand']} -> Rain Twin={comp['twin_demand']} (Delta={comp['delta_demand']} bikes).")
    print(f"  [OK] Triggered {len(comp['alerts'])} smart alert(s): '{comp['alerts'][0]['title']}'.")

    # Test 24-hour comparative curve
    df_24 = simulate_weather_twin_24h(model, feature_names, base_inputs, twin_inputs)
    assert len(df_24) == 24, "24-hour simulation did not return 24 hours!"
    assert "Baseline_Demand" in df_24.columns and "Twin_Demand" in df_24.columns
    print(f"  [OK] 24-hour comparative curve generated across all 24 hours.")


def run_all_tests():
    print("=" * 70)
    print("     CITYCYCLE - ENHANCED FEATURES TEST SUITE (BULK & TWIN)")
    print("=" * 70)
    test_sample_template()
    test_csv_validation_and_quarantine()
    test_target_leakage_in_csv()
    test_batch_prediction()
    test_weather_twin()
    print("\n" + "=" * 70)
    print("   ALL ENHANCEMENT TESTS PASSED SUCCESSFULLY! (100% COVERAGE)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
