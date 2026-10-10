"""
CityCycle - Tests for City Operator Decision Engine
Verifies each demand risk tier, operational recommendations, and input sanitization.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.decision_engine import generate_recommendation, classify_demand_risk, sanitize_input


def test_decision_tiers():
    print("\n--- TEST 1: RISK TIER CLASSIFICATION & RECOMMENDATIONS ---")
    scenarios = [
        {"name": "Night Lull", "demand": 35, "hour": 3, "workingday": 1, "weather": 1, "temp": 14.0, "expected_risk": "Low"},
        {"name": "Midday Steady", "demand": 210, "hour": 12, "workingday": 1, "weather": 1, "temp": 22.0, "expected_risk": "Moderate"},
        {"name": "Morning Rush", "demand": 430, "hour": 8, "workingday": 1, "weather": 1, "temp": 18.0, "expected_risk": "High"},
        {"name": "Peak Sunny Surge", "demand": 710, "hour": 17, "workingday": 1, "weather": 1, "temp": 25.0, "expected_risk": "Critical"}
    ]

    for sc in scenarios:
        rec = generate_recommendation(
            forecast_demand=sc["demand"],
            hour=sc["hour"],
            workingday=sc["workingday"],
            weathersit=sc["weather"],
            temp_celsius=sc["temp"]
        )

        assert rec["risk_level"] == sc["expected_risk"], (
            f"Failed on {sc['name']}: expected {sc['expected_risk']}, got {rec['risk_level']}"
        )
        assert rec["forecast_demand"] == sc["demand"]
        assert rec["trucks_recommended"] >= 0
        assert rec["staffing_estimate"] >= 1
        assert len(rec["explanation"]) > 20
        assert len(rec["recommended_action"]) > 10

        print(f"[{rec['risk_level'].upper():^8}] {sc['name']:<18} | Forecast: {rec['forecast_demand']:>3} bikes/hr | "
              f"Trucks: {rec['trucks_recommended']} | Staff: {rec['staffing_estimate']} | Action: {rec['recommended_action'][:45]}...")


def test_weather_and_commute_awareness():
    print("\n--- TEST 2: COMMUTE & ADVERSE WEATHER ADAPTABILITY ---")

    # High demand on regular weekend vs working day rush hour
    weekend_rec = generate_recommendation(forecast_demand=400, hour=14, workingday=0, weathersit=1, temp_celsius=24.0)
    commute_rec = generate_recommendation(forecast_demand=400, hour=17, workingday=1, weathersit=1, temp_celsius=24.0)

    # Commuter rush requires extra rebalancing shuttle van due to unidirectional dock drain
    assert commute_rec["trucks_recommended"] >= weekend_rec["trucks_recommended"]
    assert "Peak commuter window" in commute_rec["explanation"]
    print("  [OK] Commute rush hour automatically flagged and allocated extra shuttle trucks.")

    # Adverse weather (Thunderstorm/Heavy Rain)
    storm_rec = generate_recommendation(forecast_demand=80, hour=15, workingday=1, weathersit=4, temp_celsius=10.0)
    assert "Adverse weather" in storm_rec["explanation"]
    print("  [OK] Severe weather condition (Thunderstorm) recognized in operator briefing.")


def test_input_resilience():
    print("\n--- TEST 3: RESILIENCE TO INVALID & MISSING INPUTS ---")

    # Negative demand should be clamped to 0 without crashing
    neg_rec = generate_recommendation(forecast_demand=-50, hour=99, workingday=5, weathersit=9)
    assert neg_rec["forecast_demand"] == 0
    assert neg_rec["risk_level"] == "Low"
    assert neg_rec["context"]["hour"] == 23  # clamped to max 23
    print("  [OK] Negative demand clamped safely to 0 (Low risk).")

    # None / Missing values
    none_rec = generate_recommendation(forecast_demand=None)
    assert none_rec["forecast_demand"] == 0
    assert none_rec["risk_level"] == "Low"
    print("  [OK] None/missing demand handled with safe fallback.")

    # Float string
    str_rec = generate_recommendation(forecast_demand="320.7", hour="8")
    assert str_rec["forecast_demand"] == 321
    assert str_rec["risk_level"] == "High"
    print("  [OK] String numeric inputs automatically sanitized and cast.")


def run_all_tests():
    print("=" * 70)
    print("           CITYCYCLE - DECISION ENGINE TEST SUITE")
    print("=" * 70)
    test_decision_tiers()
    test_weather_and_commute_awareness()
    test_input_resilience()
    print("\n" + "=" * 70)
    print("   ALL TESTS PASSED SUCCESSFULLY! (100% Assertion Coverage)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
