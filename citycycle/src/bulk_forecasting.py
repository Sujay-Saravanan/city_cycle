"""
CityCycle - Bulk Forecasting and CSV Processing Module
Handles sample template generation, dual-schema validation (Official UCI and CityCycle format),
unit conversion, target leakage quarantining, and batch predictions through the trained Random Forest model.
"""

import io
import math
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from src.decision_engine import generate_recommendation
from src.feature_engineering import FEATURE_COLUMNS

# Standard required calendar/time fields common to both schemas
CORE_CALENDAR_COLUMNS = [
    "hr",
    "workingday",
    "season",
    "mnth",
    "weekday",
    "weathersit"
]

# Schema A: Official UCI Benchmark format
UCI_WEATHER_COLUMNS = ["temp", "hum", "windspeed"]

# Schema B: CityCycle converted format
CITYCYCLE_WEATHER_COLUMNS = ["temp_celsius", "humidity_pct", "windspeed_kmh"]

# Backward compatibility alias for CityCycle template
REQUIRED_COLUMNS = CORE_CALENDAR_COLUMNS + CITYCYCLE_WEATHER_COLUMNS

# Target leakage columns: strictly prohibited from model feature inputs
LEAKAGE_COLUMNS = ["casual", "registered", "cnt", "instant", "dteday"]


def generate_sample_template() -> pd.DataFrame:
    """Generates a realistic 12-hour sample dataset with CityCycle converted headers."""
    sample_data = [
        {"record_id": "REC-001", "hr": 7, "temp_celsius": 18.5, "humidity_pct": 65, "windspeed_kmh": 12.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-002", "hr": 8, "temp_celsius": 20.0, "humidity_pct": 60, "windspeed_kmh": 10.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-003", "hr": 9, "temp_celsius": 22.0, "humidity_pct": 55, "windspeed_kmh": 14.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-004", "hr": 12, "temp_celsius": 26.5, "humidity_pct": 48, "windspeed_kmh": 16.0, "weathersit": 2, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-005", "hr": 14, "temp_celsius": 28.0, "humidity_pct": 45, "windspeed_kmh": 18.0, "weathersit": 2, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-006", "hr": 17, "temp_celsius": 25.0, "humidity_pct": 52, "windspeed_kmh": 15.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-007", "hr": 18, "temp_celsius": 23.5, "humidity_pct": 58, "windspeed_kmh": 12.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-008", "hr": 19, "temp_celsius": 21.0, "humidity_pct": 65, "windspeed_kmh": 9.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-009", "hr": 21, "temp_celsius": 19.0, "humidity_pct": 72, "windspeed_kmh": 8.0, "weathersit": 2, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-010", "hr": 23, "temp_celsius": 16.5, "humidity_pct": 80, "windspeed_kmh": 7.0, "weathersit": 2, "workingday": 1, "season": 2, "mnth": 6, "weekday": 1, "holiday": 0},
        {"record_id": "REC-011", "hr": 2, "temp_celsius": 14.0, "humidity_pct": 85, "windspeed_kmh": 6.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 2, "holiday": 0},
        {"record_id": "REC-012", "hr": 5, "temp_celsius": 13.5, "humidity_pct": 88, "windspeed_kmh": 5.0, "weathersit": 1, "workingday": 1, "season": 2, "mnth": 6, "weekday": 2, "holiday": 0}
    ]
    return pd.DataFrame(sample_data)


def detect_dataset_schema(columns: List[str]) -> str:
    """Detects whether the uploaded dataset is in UCI format or CityCycle format."""
    has_uci = all(col in columns for col in UCI_WEATHER_COLUMNS)
    has_citycycle = all(col in columns for col in CITYCYCLE_WEATHER_COLUMNS)

    if has_uci:
        return "UCI"
    elif has_citycycle:
        return "CITYCYCLE"
    return "UNKNOWN"


def validate_and_filter_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Validates uploaded CSV structure, recognizes official UCI or CityCycle schemas,
    converts normalized UCI values to standard units, enforces physical bounds,
    quarantines invalid rows with itemized reasons, and preserves user's original columns.

    Returns: (valid_df, invalid_df, error_messages)
    """
    error_messages = []

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), ["Uploaded file is empty (0 rows)."]

    # Operate on a separate working copy so user's original data is never mutated
    working_df = df.copy()

    # Detect schema
    schema = detect_dataset_schema(working_df.columns.tolist())

    # Check calendar fields
    missing_calendar = [col for col in CORE_CALENDAR_COLUMNS if col not in working_df.columns]
    if missing_calendar:
        error_messages.append(f"Missing required calendar columns: {', '.join(missing_calendar)}")
        return pd.DataFrame(), working_df, error_messages

    if schema == "UNKNOWN":
        error_messages.append(
            "Unrecognized weather schema. The file must contain either official UCI columns "
            f"({', '.join(UCI_WEATHER_COLUMNS)}) or converted columns ({', '.join(CITYCYCLE_WEATHER_COLUMNS)})."
        )
        return pd.DataFrame(), working_df, error_messages

    # Informational notice and quarantine for rider subcategory target leakage
    if "cnt" in working_df.columns:
        error_messages.append(
            "Evaluation target detected ('cnt'). Retained in results for ground-truth performance evaluation, "
            "but strictly quarantined from model prediction inputs."
        )
    for col in ["casual", "registered"]:
        if col in working_df.columns:
            error_messages.append(
                f"Rider subcategory column ('{col}') quarantined and excluded from prediction features to eliminate target leakage."
            )
            working_df = working_df.drop(columns=[col])

    # Apply UCI Unit Conversions if in UCI Format:
    # - temp_celsius = temp * 47 - 8
    # - humidity_pct = hum * 100
    # - windspeed_kmh = windspeed * 67
    if schema == "UCI":
        working_df["temp_celsius"] = np.round(working_df["temp"].astype(float) * 47.0 - 8.0, 2)
        working_df["humidity_pct"] = np.round(working_df["hum"].astype(float) * 100.0, 1)
        working_df["windspeed_kmh"] = np.round(working_df["windspeed"].astype(float) * 67.0, 2)
        if "atemp" not in working_df.columns:
            working_df["atemp"] = np.clip(working_df["temp"].astype(float) + 0.02, 0.0, 1.0)
    else:  # CityCycle Format
        # Provide normalized equivalents for model consumption
        working_df["temp"] = np.clip((working_df["temp_celsius"].astype(float) + 8.0) / 47.0, 0.0, 1.0)
        working_df["hum"] = np.clip(working_df["humidity_pct"].astype(float) / 100.0, 0.0, 1.0)
        working_df["windspeed"] = np.clip(working_df["windspeed_kmh"].astype(float) / 67.0, 0.0, 1.0)
        working_df["atemp"] = np.clip(working_df["temp"] + 0.02, 0.0, 1.0)

    # Defaults for optional metadata
    if "holiday" not in working_df.columns:
        working_df["holiday"] = 0
    if "yr" not in working_df.columns:
        working_df["yr"] = 1
    if "record_id" not in working_df.columns:
        if "instant" in working_df.columns:
            working_df["record_id"] = working_df["instant"].astype(str)
        else:
            working_df["record_id"] = [f"ROW-{i+1:05d}" for i in range(len(working_df))]

    valid_indices = []
    invalid_indices = []

    for idx, row in working_df.iterrows():
        row_id = row.get("record_id", f"Row {idx+1}")
        issues = []

        # Validate Hour (0-23)
        try:
            hr = int(row["hr"])
            if not (0 <= hr <= 23):
                issues.append("hr must be 0-23")
        except (ValueError, TypeError):
            issues.append("hr is non-integer")

        # Validate Weather Situation (1-4)
        try:
            weather = int(row["weathersit"])
            if weather not in [1, 2, 3, 4]:
                issues.append("weathersit must be 1, 2, 3, or 4")
        except (ValueError, TypeError):
            issues.append("weathersit is invalid")

        # Validate Working Day (0 or 1)
        try:
            wkday = int(row["workingday"])
            if wkday not in [0, 1]:
                issues.append("workingday must be 0 or 1")
        except (ValueError, TypeError):
            issues.append("workingday must be 0 or 1")

        # Validate Season (1-4)
        try:
            season = int(row["season"])
            if season not in [1, 2, 3, 4]:
                issues.append("season must be 1-4")
        except (ValueError, TypeError):
            issues.append("season is invalid")

        # Validate Month (1-12)
        try:
            mnth = int(row["mnth"])
            if not (1 <= mnth <= 12):
                issues.append("mnth must be 1-12")
        except (ValueError, TypeError):
            issues.append("mnth is invalid")

        # Validate Weekday (0-6)
        try:
            weekday = int(row["weekday"])
            if not (0 <= weekday <= 6):
                issues.append("weekday must be 0-6")
        except (ValueError, TypeError):
            issues.append("weekday is invalid")

        # Validate Temperature Range
        try:
            temp_c = float(row["temp_celsius"])
            if not (-30.0 <= temp_c <= 60.0):
                issues.append("temp_celsius out of physical range (-30 to 60 C)")
        except (ValueError, TypeError):
            issues.append("temp_celsius is non-numeric")

        # Validate Humidity Range
        try:
            hum = float(row["humidity_pct"])
            if not (0.0 <= hum <= 100.0):
                issues.append("humidity_pct out of range (0-100%)")
        except (ValueError, TypeError):
            issues.append("humidity_pct is non-numeric")

        # Validate Wind Speed Range
        try:
            wind = float(row["windspeed_kmh"])
            if not (0.0 <= wind <= 120.0):
                issues.append("windspeed_kmh out of range (0-120 km/h)")
        except (ValueError, TypeError):
            issues.append("windspeed_kmh is non-numeric")

        if issues:
            if len(invalid_indices) < 10:  # Log first 10 for clarity
                error_messages.append(f"{row_id}: {'; '.join(issues)}")
            invalid_indices.append(idx)
        else:
            valid_indices.append(idx)

    if len(invalid_indices) > 10:
        error_messages.append(f"... and {len(invalid_indices) - 10} additional invalid records.")

    valid_df = working_df.loc[valid_indices].reset_index(drop=True)
    invalid_df = working_df.loc[invalid_indices].reset_index(drop=True)

    return valid_df, invalid_df, error_messages


def predict_batch(model, feature_names: List[str], valid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms valid records into the exact 17 features expected by bike_demand_model.pkl,
    generates demand predictions in a single high-performance vectorized call,
    calculates operational recommendations, and computes prediction residuals if 'cnt' is present.
    """
    if valid_df.empty:
        return pd.DataFrame()

    n = len(valid_df)
    hr = valid_df["hr"].astype(int).values
    workingday = valid_df["workingday"].astype(int).values
    temp_celsius = valid_df["temp_celsius"].astype(float).values
    temp_norm = valid_df["temp"].astype(float).values
    atemp_norm = valid_df["atemp"].astype(float).values if "atemp" in valid_df.columns else np.clip(temp_norm + 0.02, 0.0, 1.0)
    hum_norm = valid_df["hum"].astype(float).values
    wind_norm = valid_df["windspeed"].astype(float).values
    weathersit = valid_df["weathersit"].astype(int).values
    season = valid_df["season"].astype(int).values
    mnth = valid_df["mnth"].astype(int).values
    weekday = valid_df["weekday"].astype(int).values
    holiday = valid_df["holiday"].astype(int).values if "holiday" in valid_df.columns else np.zeros(n, dtype=int)
    yr = valid_df["yr"].astype(int).values if "yr" in valid_df.columns else np.ones(n, dtype=int)

    # Engineered features
    is_rush_hour = ((np.isin(hr, [7, 8, 9, 17, 18, 19])) & (workingday == 1)).astype(int)
    is_night = np.isin(hr, [0, 1, 2, 3, 4, 5]).astype(int)
    sin_hr = np.sin(2 * np.pi * hr / 24.0)
    cos_hr = np.cos(2 * np.pi * hr / 24.0)

    # Assemble 17 features matrix
    feat_matrix = pd.DataFrame({
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp_norm,
        "atemp": atemp_norm,
        "hum": hum_norm,
        "windspeed": wind_norm,
        "is_rush_hour": is_rush_hour,
        "is_night": is_night,
        "sin_hr": sin_hr,
        "cos_hr": cos_hr,
        "temp_celsius": temp_celsius
    })[feature_names]

    # Vectorized model prediction
    raw_preds = model.predict(feat_matrix)
    pred_demands = np.maximum(0, np.round(raw_preds)).astype(int)

    # Attach predictions and operational decision outputs
    result_df = valid_df.copy()
    result_df["predicted_demand"] = pred_demands

    risk_levels = []
    vans = []
    staff = []
    actions = []

    for p, h, w, weath, t in zip(pred_demands, hr, workingday, weathersit, temp_celsius):
        rec = generate_recommendation(
            forecast_demand=p,
            hour=h,
            workingday=w,
            weathersit=weath,
            temp_celsius=t
        )
        risk_levels.append(rec["risk_level"])
        vans.append(rec["trucks_recommended"])
        staff.append(rec["staffing_estimate"])
        actions.append(rec["recommended_action"])

    result_df["risk_level"] = risk_levels
    result_df["recommended_vans"] = vans
    result_df["recommended_staff"] = staff
    result_df["action_directive"] = actions

    # Calculate error metrics if ground-truth 'cnt' is present
    if "cnt" in result_df.columns and result_df["cnt"].notnull().any():
        actual = result_df["cnt"].astype(float).values
        result_df["prediction_error"] = np.round(pred_demands - actual).astype(int)
        result_df["abs_error"] = np.round(np.abs(pred_demands - actual)).astype(int)

    return result_df
