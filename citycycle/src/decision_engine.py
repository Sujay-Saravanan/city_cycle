"""
CityCycle - City Operator Decision Engine
Translates hourly bike demand forecasts into actionable operational recommendations
for fleet rebalancing vans and field staffing.

Note: The operational rules below are configurable heuristic models designed
for simulation and hackathon demonstration purposes.
"""

from typing import Dict, Any, Optional
import math


# Configurable Demand Thresholds (bikes/hour)
THRESHOLDS = {
    "low_max": 100,       # 0 - 99: Low
    "moderate_max": 300,  # 100 - 299: Moderate
    "high_max": 550       # 300 - 549: High, >= 550: Critical
}

# Weather severity map (UCI Bike Sharing dataset codes)
WEATHER_DESCRIPTIONS = {
    1: "Clear / Partly Cloudy",
    2: "Mist / Overcast",
    3: "Light Rain / Snow",
    4: "Heavy Rain / Thunderstorm"
}


def sanitize_input(value: Any, default: float = 0.0, min_val: float = 0.0, max_val: Optional[float] = None) -> float:
    """Safely converts and bounds numeric inputs, handling None, NaN, and invalid strings."""
    try:
        if value is None:
            return default
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        if min_val is not None and val < min_val:
            val = min_val
        if max_val is not None and val > max_val:
            val = max_val
        return val
    except (ValueError, TypeError):
        return default


def classify_demand_risk(demand: float) -> str:
    """Classifies hourly forecast demand into operational risk tiers."""
    if demand < THRESHOLDS["low_max"]:
        return "Low"
    elif demand < THRESHOLDS["moderate_max"]:
        return "Moderate"
    elif demand < THRESHOLDS["high_max"]:
        return "High"
    else:
        return "Critical"


def calculate_staffing_estimate(demand: float, hour: int, workingday: int, weathersit: int) -> int:
    """
    Estimates recommended field staff using configurable heuristic rules:
    - Base coverage: 1 staff (night) or 2 staff (day)
    - Additional field capacity: +1 staff per 120 riders above 100
    - Rush hour adjustment on working days: +1 coordinator
    """
    is_night = hour in [0, 1, 2, 3, 4, 5]
    base_staff = 1 if is_night else 2

    # Scale with demand volume
    demand_staff = int(max(0, demand - 100) // 120)

    # Rush hour surge requirement on working days
    rush_adj = 1 if (workingday == 1 and hour in [7, 8, 9, 17, 18, 19] and demand >= 300) else 0

    # Severe weather reduces outdoor bicycle movement
    weather_adj = -1 if (weathersit in [3, 4] and demand < 200) else 0

    total_staff = max(1, min(8, base_staff + demand_staff + rush_adj + weather_adj))
    return int(total_staff)


def calculate_rebalancing_trucks(demand: float, risk_level: str, hour: int, workingday: int) -> int:
    """
    Calculates recommended redistribution vans/trucks to keep station docks balanced:
    - Low: 0 (No active shifts needed)
    - Moderate: 1 (Routine rebalancing loop)
    - High: 2 to 3 (Active commuter corridor shuttle)
    - Critical: 4+ (Emergency all-hands fleet redistribution)
    """
    if risk_level == "Low":
        return 0
    elif risk_level == "Moderate":
        return 1
    elif risk_level == "High":
        # Morning and evening working rush hours have asymmetric dock drain
        is_rush = workingday == 1 and hour in [7, 8, 9, 17, 18, 19]
        return 3 if is_rush else 2
    else:  # Critical
        return 4 if demand < 700 else 5


def generate_explanation(
    demand: float,
    risk_level: str,
    hour: int,
    workingday: int,
    weathersit: int,
    temp_celsius: float,
    trucks: int,
    staff: int
) -> str:
    """Constructs a clear, non-technical explanation for city operations teams."""
    weather_desc = WEATHER_DESCRIPTIONS.get(int(weathersit), "Standard weather")
    day_type = "working weekday" if workingday == 1 else "weekend/holiday"
    is_commute = workingday == 1 and hour in [7, 8, 9, 17, 18, 19]

    reasons = [f"Forecast demand is {int(demand)} bikes/hour at {hour:02d}:00 on a {day_type}."]

    if is_commute:
        reasons.append("Peak commuter window creates asymmetric traffic flow between transit hubs and residential docks.")
    if weathersit in [3, 4]:
        reasons.append(f"Adverse weather ({weather_desc}) dampens casual trips and requires rain-safe equipment handling.")
    elif temp_celsius >= 20.0 and weathersit == 1:
        reasons.append(f"Pleasant temperature ({temp_celsius:.1f}°C) and clear skies stimulate high ridership.")

    if risk_level == "Low":
        reasons.append(f"Minimal station depletion risk. Recommended: {staff} technician for maintenance/battery swaps.")
    elif risk_level == "Moderate":
        reasons.append(f"Standard traffic volume. Recommended: {trucks} van on routine rounds and {staff} field staff.")
    elif risk_level == "High":
        reasons.append(f"High stockout risk at primary stations. Deploy {trucks} redistribution vans and {staff} staff.")
    else:
        reasons.append(f"CRITICAL SURGE: Rapid dock exhaustion expected. Mobilize {trucks} emergency vans and {staff} staff immediately.")

    return " ".join(reasons)


def generate_recommendation(
    forecast_demand: float,
    hour: int = 12,
    workingday: int = 1,
    weathersit: int = 1,
    temp_celsius: float = 20.0
) -> Dict[str, Any]:
    """
    Main decision engine function.
    Converts forecast demand and operational context into a structured recommendation.
    """
    # 1. Sanitize inputs safely
    clean_demand = sanitize_input(forecast_demand, default=0.0, min_val=0.0)
    clean_hour = int(sanitize_input(hour, default=12, min_val=0, max_val=23))
    clean_workingday = 1 if int(sanitize_input(workingday, default=1, min_val=0, max_val=1)) == 1 else 0
    clean_weathersit = int(sanitize_input(weathersit, default=1, min_val=1, max_val=4))
    clean_temp = sanitize_input(temp_celsius, default=20.0, min_val=-20.0, max_val=50.0)

    # 2. Risk classification
    risk_level = classify_demand_risk(clean_demand)

    # 3. Operations calculations
    trucks = calculate_rebalancing_trucks(clean_demand, risk_level, clean_hour, clean_workingday)
    staff = calculate_staffing_estimate(clean_demand, clean_hour, clean_workingday, clean_weathersit)

    # 4. Action statement
    actions = {
        "Low": "Station docks stable. Perform scheduled bike inspections and battery charging.",
        "Moderate": "Maintain routine rebalancing loop. Monitor dock levels at transit intersections.",
        "High": "Dispatch active redistribution vans to prevent residential and office dock stockouts.",
        "Critical": "URGENT REBALANCING: Deploy maximum shuttle vans to relocate bikes and prevent network gridlock."
    }
    recommended_action = actions[risk_level]

    # 5. Natural explanation
    explanation = generate_explanation(
        demand=clean_demand,
        risk_level=risk_level,
        hour=clean_hour,
        workingday=clean_workingday,
        weathersit=clean_weathersit,
        temp_celsius=clean_temp,
        trucks=trucks,
        staff=staff
    )

    return {
        "forecast_demand": int(round(clean_demand)),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "trucks_recommended": trucks,
        "staffing_estimate": staff,
        "context": {
            "hour": clean_hour,
            "workingday": clean_workingday,
            "weather_condition": WEATHER_DESCRIPTIONS.get(clean_weathersit, "Clear"),
            "temp_celsius": round(clean_temp, 1)
        },
        "explanation": explanation
    }
