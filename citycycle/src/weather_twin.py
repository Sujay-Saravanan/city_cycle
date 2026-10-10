"""
CityCycle - Weather Impact Twin and Smart Weather Alerts Module
Compares baseline vs simulated weather scenarios across single hours and full 24-hour cycles,
and generates operational alerts for city bike-sharing operators.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.decision_engine import generate_recommendation

TWIN_PRESETS = {
    "🌧️ Heavy Rain / Thunderstorm": {
        "weathersit": 4,
        "temp_offset": -3.5,
        "humidity_pct": 92.0,
        "windspeed_kmh": 28.0,
        "description": "Heavy precipitation and thunderstorm causing widespread trip cancellation."
    },
    "☀️ Summer Heatwave": {
        "weathersit": 1,
        "temp_override": 38.0,
        "humidity_pct": 35.0,
        "windspeed_kmh": 12.0,
        "description": "Excessive afternoon heat suppressing midday ridership."
    },
    "❄️ Winter Cold Snap": {
        "weathersit": 2,
        "temp_override": -2.0,
        "humidity_pct": 75.0,
        "windspeed_kmh": 32.0,
        "description": "Sub-zero chill with brisk winds dampening casual cycling."
    },
    "💨 High Wind Advisory": {
        "weathersit": 2,
        "temp_offset": -1.0,
        "humidity_pct": 60.0,
        "windspeed_kmh": 46.0,
        "description": "Sustained strong winds impacting rider stability on bridges and corridors."
    }
}


def build_feature_row(inputs: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    """Builds the 17-element model feature vector with zero target leakage."""
    hr = int(inputs["hr"])
    workingday = int(inputs["workingday"])
    temp_c = float(inputs["temp_celsius"])

    is_rush_hour = 1 if (hr in [7, 8, 9, 17, 18, 19] and workingday == 1) else 0
    is_night = 1 if hr in [0, 1, 2, 3, 4, 5] else 0
    sin_hr = np.sin(2 * np.pi * hr / 24.0)
    cos_hr = np.cos(2 * np.pi * hr / 24.0)
    temp_norm = np.clip(temp_c / 41.0, 0.0, 1.0)
    atemp_norm = np.clip(temp_norm + 0.02, 0.0, 1.0)
    hum_norm = np.clip(float(inputs["humidity_pct"]) / 100.0, 0.0, 1.0)
    wind_norm = np.clip(float(inputs["windspeed_kmh"]) / 67.0, 0.0, 1.0)

    feat_dict = {
        "season": int(inputs.get("season", 2)),
        "yr": 1,
        "mnth": int(inputs.get("mnth", 6)),
        "hr": hr,
        "holiday": int(inputs.get("holiday", 0)),
        "weekday": int(inputs.get("weekday", 2)),
        "workingday": workingday,
        "weathersit": int(inputs["weathersit"]),
        "temp": temp_norm,
        "atemp": atemp_norm,
        "hum": hum_norm,
        "windspeed": wind_norm,
        "is_rush_hour": is_rush_hour,
        "is_night": is_night,
        "sin_hr": sin_hr,
        "cos_hr": cos_hr,
        "temp_celsius": temp_c
    }
    return pd.DataFrame([feat_dict])[feature_names]


def generate_smart_weather_alerts(
    baseline_demand: int,
    twin_demand: int,
    baseline_rec: Dict[str, Any],
    twin_rec: Dict[str, Any],
    scenario_name: str,
    twin_inputs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Evaluates whether the weather scenario triggers actionable operational alerts."""
    alerts = []
    delta = twin_demand - baseline_demand
    pct_change = (delta / max(1, baseline_demand)) * 100.0

    # 1. Severe Weather Equipment & Safety Warning
    weathersit = int(twin_inputs.get("weathersit", 1))
    windspeed = float(twin_inputs.get("windspeed_kmh", 0.0))

    if weathersit in [3, 4]:
        alerts.append({
            "level": "warning",
            "type": "Safety & Asset Protection",
            "title": "Rain & Wet Road Conditions Detected",
            "message": f"Precipitation (Weather code {weathersit}) reduces tire traction and road visibility. "
                       "Inspect e-bike battery terminal seals; ensure field crews wear high-visibility rain gear.",
            "is_simulation": True
        })
    elif windspeed >= 40.0:
        alerts.append({
            "level": "warning",
            "type": "Safety & Asset Protection",
            "title": "High Wind Speed Advisory",
            "message": f"Sustained wind speed of {windspeed:.1f} km/h poses balance hazards for casual riders. "
                       "Anchor open station repair stands and avoid tall-van highway transport.",
            "is_simulation": True
        })

    # 2. Demand Slump Alert
    if delta <= -40 and pct_change <= -20.0:
        alerts.append({
            "level": "info",
            "type": "Fleet Optimization Alert",
            "title": f"Projected Demand Slump ({delta:+d} bikes / {pct_change:.1f}%)",
            "message": f"Simulated {scenario_name} sharply curtails ridership. "
                       f"Stand down {max(0, baseline_rec['trucks_recommended'] - twin_rec['trucks_recommended'])} redistribution vans "
                       "to save fuel; redirect available technicians to workshop bike repairs.",
            "is_simulation": True
        })

    # 3. Surge Escalation Alert
    elif delta >= 50 or (twin_rec["risk_level"] in ["High", "Critical"] and baseline_rec["risk_level"] in ["Low", "Moderate"]):
        alerts.append({
            "level": "critical",
            "type": "Surge Dispatch Alert",
            "title": f"Demand Surge Alert ({delta:+d} bikes / {pct_change:+.1f}%)",
            "message": f"Favorable riding conditions push operational risk to {twin_rec['risk_level'].upper()}. "
                       f"Deploy {twin_rec['trucks_recommended']} vans immediately to prevent station dock depletion.",
            "is_simulation": True
        })

    # 4. No Significant Variance
    if not alerts:
        alerts.append({
            "level": "nominal",
            "type": "Operations Nominal",
            "title": "Minor Weather Sensitivity",
            "message": f"Demand variation ({delta:+d} bikes / {pct_change:+.1f}%) is within standard operational tolerance. "
                       "Maintain standard scheduled patrol routes.",
            "is_simulation": True
        })

    return alerts


def simulate_weather_twin_single(
    model,
    feature_names: List[str],
    base_inputs: Dict[str, Any],
    twin_inputs: Dict[str, Any],
    scenario_name: str
) -> Dict[str, Any]:
    """Computes comparison between baseline and twin scenario for a single hour."""
    # Predict baseline
    base_feats = build_feature_row(base_inputs, feature_names)
    base_pred = max(0, int(round(model.predict(base_feats)[0])))

    # Predict twin
    twin_feats = build_feature_row(twin_inputs, feature_names)
    twin_pred = max(0, int(round(model.predict(twin_feats)[0])))

    # Decision Engine outputs
    base_rec = generate_recommendation(
        forecast_demand=base_pred,
        hour=base_inputs["hr"],
        workingday=base_inputs["workingday"],
        weathersit=base_inputs["weathersit"],
        temp_celsius=base_inputs["temp_celsius"]
    )

    twin_rec = generate_recommendation(
        forecast_demand=twin_pred,
        hour=twin_inputs["hr"],
        workingday=twin_inputs["workingday"],
        weathersit=twin_inputs["weathersit"],
        temp_celsius=twin_inputs["temp_celsius"]
    )

    delta_demand = twin_pred - base_pred
    pct_change = (delta_demand / max(1, base_pred)) * 100.0

    alerts = generate_smart_weather_alerts(
        baseline_demand=base_pred,
        twin_demand=twin_pred,
        baseline_rec=base_rec,
        twin_rec=twin_rec,
        scenario_name=scenario_name,
        twin_inputs=twin_inputs
    )

    return {
        "baseline_demand": base_pred,
        "twin_demand": twin_pred,
        "delta_demand": delta_demand,
        "pct_change": round(pct_change, 1),
        "baseline_risk": base_rec["risk_level"],
        "twin_risk": twin_rec["risk_level"],
        "baseline_vans": base_rec["trucks_recommended"],
        "twin_vans": twin_rec["trucks_recommended"],
        "baseline_staff": base_rec["staffing_estimate"],
        "twin_staff": twin_rec["staffing_estimate"],
        "baseline_action": base_rec["recommended_action"],
        "twin_action": twin_rec["recommended_action"],
        "alerts": alerts
    }


def simulate_weather_twin_24h(
    model,
    feature_names: List[str],
    base_inputs: Dict[str, Any],
    twin_inputs: Dict[str, Any]
) -> pd.DataFrame:
    """Simulates both Baseline and Weather Twin across all 24 hours of the day."""
    hours = list(range(24))
    base_curve = []
    twin_curve = []
    base_risks = []
    twin_risks = []

    for h in hours:
        # Clone inputs with active hour
        b_in = base_inputs.copy()
        b_in["hr"] = h
        b_pred = max(0, int(round(model.predict(build_feature_row(b_in, feature_names))[0])))
        base_curve.append(b_pred)
        b_rec = generate_recommendation(b_pred, hour=h, workingday=b_in["workingday"], weathersit=b_in["weathersit"], temp_celsius=b_in["temp_celsius"])
        base_risks.append(b_rec["risk_level"])

        t_in = twin_inputs.copy()
        t_in["hr"] = h
        t_pred = max(0, int(round(model.predict(build_feature_row(t_in, feature_names))[0])))
        twin_curve.append(t_pred)
        t_rec = generate_recommendation(t_pred, hour=h, workingday=t_in["workingday"], weathersit=t_in["weathersit"], temp_celsius=t_in["temp_celsius"])
        twin_risks.append(t_rec["risk_level"])

    return pd.DataFrame({
        "Hour": hours,
        "Baseline_Demand": base_curve,
        "Twin_Demand": twin_curve,
        "Delta": np.array(twin_curve) - np.array(base_curve),
        "Baseline_Risk": base_risks,
        "Twin_Risk": twin_risks
    })
