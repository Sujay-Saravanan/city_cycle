"""
CityCycle - Tests for Dataset Selector, Catalog Management & Client Model Engine
Verifies dataset catalog initialization, switching between datasets,
dynamic profiling, model compatibility checking, stale cache isolation,
and client-specific model training without touching the pretrained model.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset_manager import (
    DatasetManager,
    profile_dataset,
    check_model_compatibility,
    train_client_model
)
from src.bulk_forecasting import (
    generate_sample_template,
    predict_batch
)
from src.feature_engineering import FEATURE_COLUMNS
import joblib


def test_default_catalog_initialization():
    print("\n--- TEST 1: CATALOG INITIALIZATION WITH DEFAULT UCI BENCHMARK ---")
    dm = DatasetManager()
    options = dm.get_dataset_options()

    assert "default_uci" in dm.catalog, "default_uci not found in catalog!"
    assert dm.active_id == "default_uci", f"Default active_id should be 'default_uci', got '{dm.active_id}'"

    profile = dm.get_active_profile()
    assert profile["total_rows"] == 17379, f"Expected 17,379 rows, got {profile['total_rows']}"
    assert profile["valid_rows"] == 17379, f"Expected 17,379 valid rows, got {profile['valid_rows']}"
    assert profile["schema"] == "UCI", f"Expected UCI schema, got {profile['schema']}"
    assert profile["target_col"] == "cnt", f"Expected 'cnt' target, got {profile['target_col']}"
    assert profile["is_compatible"] is True, "Default UCI dataset should be fully compatible with pretrained model!"
    assert len(profile["missing_features"]) == 0

    print(f"  [OK] Default UCI dataset loaded: {profile['total_rows']:,} rows, target='{profile['target_col']}', compatible=True.")


def test_register_and_switch_datasets():
    print("\n--- TEST 2: DATASET REGISTRATION AND ACTIVE SELECTION SWITCHING ---")
    dm = DatasetManager()

    # Create a synthetic client dataset
    client_df = generate_sample_template()
    client_id = dm.register_client_dataset("Client Fleet - Metro North", client_df)

    assert client_id in dm.catalog, "Registered dataset not in catalog!"
    assert dm.active_id == client_id, "Active ID should point to newly registered dataset"

    active_profile = dm.get_active_profile()
    assert active_profile["name"] == "Client Fleet - Metro North"
    assert active_profile["total_rows"] == len(client_df)
    assert active_profile["schema"] == "CITYCYCLE"

    # Switch back to default
    dm.set_active_dataset("default_uci")
    assert dm.active_id == "default_uci"
    assert dm.get_active_profile()["name"].startswith("🚲 Default UCI")

    # Switch back to client
    dm.set_active_dataset(client_id)
    assert dm.active_id == client_id
    assert dm.get_active_profile()["name"] == "Client Fleet - Metro North"

    print(f"  [OK] Successfully registered client dataset and switched active selection back and forth seamlessly.")


def test_stale_cache_and_prediction_switching():
    print("\n--- TEST 3: ISOLATED BATCH PREDICTIONS & CACHE INDEPENDENCE ---")
    dm = DatasetManager()
    model_pkg = joblib.load(os.path.join(PROJECT_ROOT, "models", "bike_demand_model.pkl"))
    model = model_pkg["model"]
    feature_names = model_pkg["feature_names"]

    # Predict on dataset 1 (UCI benchmark slice)
    uci_profile = dm.get_active_profile()
    uci_slice = uci_profile["valid_df"].head(50)
    preds_uci = predict_batch(model, feature_names, uci_slice)

    # Register and predict on dataset 2 (sample template)
    sample_df = generate_sample_template()
    dm.register_client_dataset("Metro Sample", sample_df)
    client_profile = dm.get_active_profile()
    preds_client = predict_batch(model, feature_names, client_profile["valid_df"])

    assert len(preds_uci) == 50
    assert len(preds_client) == len(sample_df)
    assert not preds_uci["predicted_demand"].equals(preds_client["predicted_demand"].head(50)), "Dataset predictions must differ!"

    print(f"  [OK] Dataset predictions are completely isolated: {len(preds_uci)} rows (UCI) vs {len(preds_client)} rows (Client).")


def test_model_compatibility_guardrails():
    print("\n--- TEST 4: COMPATIBILITY GUARDRAILS ON CUSTOM/INCOMPLETE DATASETS ---")
    # Dataset missing weather columns
    incomplete_df = pd.DataFrame({
        "hr": [8, 9, 10],
        "workingday": [1, 1, 1],
        "weathersit": [1, 1, 1],
        "season": [1, 1, 1],
        "mnth": [5, 5, 5],
        "weekday": [2, 2, 2]
        # missing temp, hum, windspeed
    })

    is_compat, missing = check_model_compatibility(incomplete_df, FEATURE_COLUMNS)
    assert is_compat is False, "Should be incompatible when weather features are missing!"
    assert any("Weather metrics" in m for m in missing), f"Expected weather metrics warning, got {missing}"

    # Dataset with empty df
    empty_compat, empty_missing = check_model_compatibility(pd.DataFrame(), FEATURE_COLUMNS)
    assert empty_compat is False

    print(f"  [OK] Guardrails correctly blocked incompatible data schema and reported: {missing[0]}")


def test_client_model_training_and_isolation():
    print("\n--- TEST 5: CLIENT-SPECIFIC MODEL TRAINING & READ-ONLY ISOLATION ---")
    pretrained_path = os.path.join(PROJECT_ROOT, "models", "bike_demand_model.pkl")
    mtime_before = os.path.getmtime(pretrained_path)

    # Use first 200 records of UCI data with 'cnt' target to simulate client training
    data_path = os.path.join(PROJECT_ROOT, "data", "hour.csv")
    train_slice = pd.read_csv(data_path).head(200)

    # Train client model (RandomForest)
    res_rf = train_client_model(train_slice, target_col="cnt", model_type="RandomForest")
    assert "model" in res_rf
    assert res_rf["metrics"]["r2"] is not None
    assert "casual" not in res_rf["feature_names"], "Leakage: 'casual' should never be in feature list!"
    assert "registered" not in res_rf["feature_names"], "Leakage: 'registered' should never be in feature list!"
    assert "cnt" not in res_rf["feature_names"], "Leakage: target 'cnt' should never be in feature list!"

    # Train client model (Ridge)
    res_ridge = train_client_model(train_slice, target_col="cnt", model_type="Ridge")
    assert "model" in res_ridge
    assert res_ridge["metrics"]["mae"] > 0

    # Verify pretrained model file was not modified
    mtime_after = os.path.getmtime(pretrained_path)
    assert mtime_before == mtime_after, "Pretrained model file must remain untouched during client model training!"

    print(f"  [OK] Client model trained successfully: RF test MAE = {res_rf['metrics']['mae']}, R2 = {res_rf['metrics']['r2']}.")
    print(f"  [OK] Target leakage strictly avoided: {res_rf['feature_names']}")
    print(f"  [OK] Pretrained model file verified intact and untouched.")


def run_all_tests():
    print("=" * 70)
    print("       CITYCYCLE - DATASET SELECTOR TEST SUITE")
    print("=" * 70)
    test_default_catalog_initialization()
    test_register_and_switch_datasets()
    test_stale_cache_and_prediction_switching()
    test_model_compatibility_guardrails()
    test_client_model_training_and_isolation()
    print("\n" + "=" * 70)
    print("   ALL DATASET SELECTOR TESTS PASSED (100% VERIFIED)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
