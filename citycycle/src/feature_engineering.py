"""
CityCycle - Feature Engineering Module
Transforms raw hourly bike data into ML-ready feature matrices with strict leakage prevention.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

# Explicit feature list to ensure reproducibility and transparency
FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "is_rush_hour",
    "is_night",
    "sin_hr",
    "cos_hr",
    "temp_celsius"
]

TARGET_COLUMN = "cnt"

# Columns strictly prohibited from training to prevent target leakage
LEAKAGE_COLUMNS = ["casual", "registered", "cnt", "instant", "dteday"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw DataFrame by engineering domain-specific features.
    Guarantees no target leakage columns are included in feature matrix.
    """
    df = df.copy()

    # Ensure chronological sort
    if "dteday" in df.columns and "hr" in df.columns:
        df = df.sort_values(by=["dteday", "hr"]).reset_index(drop=True)

    # 1. Rush hour indicator: 7-9 AM and 5-7 PM on working days
    is_commute_time = df["hr"].isin([7, 8, 9, 17, 18, 19])
    df["is_rush_hour"] = (is_commute_time & (df["workingday"] == 1)).astype(int)

    # 2. Night indicator: 12 AM to 5 AM (naturally low demand)
    df["is_night"] = df["hr"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # 3. Cyclical hour encoding (24-hour periodicity)
    # Allows model to understand that 23:00 is temporally adjacent to 00:00
    df["sin_hr"] = np.sin(2 * np.pi * df["hr"] / 24.0)
    df["cos_hr"] = np.cos(2 * np.pi * df["hr"] / 24.0)

    # 4. Human-readable Celsius temperature (UCI formula: temp_norm * 41)
    df["temp_celsius"] = df["temp"] * 41.0

    return df


def split_chronological(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Splits the data chronologically into train and test sets.
    Earlier records (80%) are used for training; latest records (20%) are used for testing.
    """
    processed_df = create_features(df)

    n_total = len(processed_df)
    n_train = int(n_total * train_ratio)

    train_df = processed_df.iloc[:n_train]
    test_df = processed_df.iloc[n_train:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test, FEATURE_COLUMNS
