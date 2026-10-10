"""
CityCycle - Smart Bike Demand Forecasting & Fleet Operations Dashboard
Interactive web application powered by Streamlit, Scikit-Learn, and Plotly.
Includes:
- Dynamic Dataset Selector & Active Dataset Profiling
- Real-Time Simulation & Operations Decision Engine
- Custom Data Prediction & Bulk CSV Forecasting (with Ground-Truth Evaluation)
- Client-Specific In-Memory Model Training
- Weather Impact Twin & Smart Weather Alerts
- Validation Benchmarks & Feature Importance Insights
"""

import io
import json
import os
import sys

# Ensure root directory is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.decision_engine import generate_recommendation
from src.bulk_forecasting import (
    generate_sample_template,
    validate_and_filter_csv,
    predict_batch
)
from src.weather_twin import (
    TWIN_PRESETS,
    simulate_weather_twin_single,
    simulate_weather_twin_24h,
    build_feature_row
)
from src.dataset_manager import (
    DatasetManager,
    train_client_model,
    FEATURE_COLUMNS
)

# Page Configuration
st.set_page_config(
    page_title="CityCycle | Bike Demand Forecasting & Fleet Operations",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .risk-badge {
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    .risk-low { background-color: #DCFCE7; color: #166534; }
    .risk-moderate { background-color: #FEF9C3; color: #854D0E; }
    .risk-high { background-color: #FFEDD5; color: #9A3412; }
    .risk-critical { background-color: #FEE2E2; color: #991B1B; }
    .alert-box {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .alert-warning { background-color: #FFFBEB; border-left: 5px solid #F59E0B; }
    .alert-critical { background-color: #FEF2F2; border-left: 5px solid #EF4444; }
    .alert-info { background-color: #EFF6FF; border-left: 5px solid #3B82F6; }
    .alert-nominal { background-color: #F0FDF4; border-left: 5px solid #22C55E; }
    .dataset-badge {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_package():
    """Loads the trained Random Forest model and evaluation metrics from disk."""
    model_path = os.path.join(PROJECT_ROOT, "models", "bike_demand_model.pkl")
    metrics_path = os.path.join(PROJECT_ROOT, "models", "evaluation_results.json")

    if not os.path.exists(model_path):
        return None, None

    model_pkg = joblib.load(model_path)
    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    return model_pkg, metrics_data


def predict_demand(model, feature_names, inputs: dict) -> float:
    """Prepares the feature vector and generates a demand prediction."""
    feature_df = build_feature_row(inputs, feature_names)
    pred = model.predict(feature_df)[0]
    return max(0.0, float(pred))


def main():
    model_pkg, metrics_data = load_model_package()

    if model_pkg is None:
        st.error("Model file `models/bike_demand_model.pkl` not found. Please run `src/train.py` first.")
        return

    model = model_pkg["model"]
    feature_names = model_pkg["feature_names"]

    # Initialize DatasetManager in Session State
    if "dataset_manager" not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()
    dm: DatasetManager = st.session_state.dataset_manager

    # Header
    st.markdown('<div class="main-header">🚲 CityCycle: Smart Bike Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Hourly Demand Forecasting, Multi-Dataset Operations & Fleet Decision Engine</div>',
        unsafe_allow_html=True
    )

    # =========================================================================
    # SIDEBAR: DATASET SELECTOR
    # =========================================================================
    st.sidebar.header("📁 Active Dataset Selector")
    dataset_options = dm.get_dataset_options()
    option_names = list(dataset_options.keys()) + ["➕ Upload New CSV Dataset..."]

    # Determine current index
    current_active_id = dm.active_id
    current_index = 0
    for idx, (name, d_id) in enumerate(dataset_options.items()):
        if d_id == current_active_id:
            current_index = idx
            break

    selected_option = st.sidebar.selectbox(
        "Choose Active Dataset:",
        options=option_names,
        index=current_index,
        help="Select which dataset is active for predictions, previews, and operations."
    )

    if selected_option == "➕ Upload New CSV Dataset...":
        sidebar_uploaded = st.sidebar.file_uploader("Upload Client CSV File", type=["csv"], key="sidebar_csv_uploader")
        if sidebar_uploaded is not None:
            try:
                new_raw_df = pd.read_csv(sidebar_uploaded)
                new_id = dm.register_client_dataset(sidebar_uploaded.name, new_raw_df)
                st.sidebar.success(f"Added {sidebar_uploaded.name} ({len(new_raw_df):,} rows)")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error parsing uploaded CSV: {e}")
    else:
        chosen_id = dataset_options[selected_option]
        if dm.active_id != chosen_id:
            dm.set_active_dataset(chosen_id)
            # Invalidate any old prediction cache to guarantee fresh state
            if "active_batch_results" in st.session_state:
                del st.session_state["active_batch_results"]
            st.rerun()

    active_profile = dm.get_active_profile()

    # Active Dataset Status Card
    compat_badge = "🟢 Pretrained Compatible" if active_profile["is_compatible"] else "⚠️ Custom Features"
    target_badge = active_profile["target_col"] if active_profile["target_col"] else "Unlabeled"

    st.sidebar.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; font-size: 0.85rem;">
        <strong>Active Dataset:</strong> {active_profile['name'][:30]}<br>
        <strong>Total Records:</strong> {active_profile['total_rows']:,}<br>
        <strong>Valid Records:</strong> {active_profile['valid_rows']:,}<br>
        <strong>Detected Target:</strong> <code>{target_badge}</code><br>
        <strong>Status:</strong> {compat_badge}
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Global Presets and Environment Controls
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Hackathon Demo Presets")
    preset = st.sidebar.selectbox(
        "Load Preconfigured Scenario:",
        [
            "Custom Scenario",
            "☀️ Sunny Morning Commute (Peak Rush)",
            "🌧️ Rainy Autumn Evening (Adverse Weather)",
            "🏖️ Sunny Weekend Afternoon (Leisure)",
            "🌙 Late Night Maintenance Window"
        ]
    )

    default_hour = 8
    default_workingday = 1
    default_weekday = 2
    default_season = 2
    default_month = 6
    default_weather = 1
    default_temp = 24.0
    default_hum = 55
    default_wind = 12.0
    default_holiday = 0

    if preset == "☀️ Sunny Morning Commute (Peak Rush)":
        default_hour = 8
        default_workingday = 1
        default_weekday = 2
        default_season = 2
        default_month = 6
        default_weather = 1
        default_temp = 24.0
        default_hum = 50
        default_wind = 10.0
    elif preset == "🌧️ Rainy Autumn Evening (Adverse Weather)":
        default_hour = 18
        default_workingday = 1
        default_weekday = 3
        default_season = 3
        default_month = 10
        default_weather = 3
        default_temp = 12.0
        default_hum = 88
        default_wind = 25.0
    elif preset == "🏖️ Sunny Weekend Afternoon (Leisure)":
        default_hour = 14
        default_workingday = 0
        default_weekday = 6
        default_season = 2
        default_month = 7
        default_weather = 1
        default_temp = 27.0
        default_hum = 45
        default_wind = 14.0
    elif preset == "🌙 Late Night Maintenance Window":
        default_hour = 3
        default_workingday = 1
        default_weekday = 1
        default_season = 1
        default_month = 4
        default_weather = 1
        default_temp = 11.0
        default_hum = 65
        default_wind = 8.0

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Simulation Parameters")

    hour = st.sidebar.slider("Hour of Day (0–23):", 0, 23, default_hour)
    temp_c = st.sidebar.slider("Temperature (°C):", -5.0, 40.0, float(default_temp), step=0.5)
    humidity = st.sidebar.slider("Relative Humidity (%):", 10, 100, int(default_hum))
    windspeed = st.sidebar.slider("Wind Speed (km/h):", 0.0, 50.0, float(default_wind), step=1.0)

    weather_labels = {
        1: "Clear / Few Clouds",
        2: "Mist / Overcast",
        3: "Light Rain / Snow",
        4: "Heavy Rain / Thunderstorm"
    }
    weather_sit = st.sidebar.selectbox(
        "Weather Condition:",
        options=[1, 2, 3, 4],
        index=default_weather - 1,
        format_func=lambda x: f"{x}: {weather_labels[x]}"
    )

    day_type = st.sidebar.radio(
        "Day Category:",
        options=["Working Day (Commute)", "Weekend / Holiday"],
        index=0 if default_workingday == 1 else 1
    )
    workingday = 1 if "Working Day" in day_type else 0

    season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    season = st.sidebar.selectbox(
        "Season:",
        options=[1, 2, 3, 4],
        index=default_season - 1,
        format_func=lambda x: season_labels[x]
    )

    month = default_month
    weekday = default_weekday if workingday == 1 else 6
    holiday = default_holiday

    # Global shared inputs dictionary
    inputs = {
        "hr": hour,
        "workingday": workingday,
        "weekday": weekday,
        "mnth": month,
        "season": season,
        "holiday": holiday,
        "weathersit": weather_sit,
        "temp_celsius": temp_c,
        "humidity_pct": humidity,
        "windspeed_kmh": windspeed
    }

    # Top-Level Navigation Tabs
    tab_sim, tab_bulk, tab_twin, tab_valid = st.tabs([
        "🎛️ Real-Time Simulator & Decision Engine",
        "📥 Custom Prediction & Bulk Forecasting",
        "🌦️ Weather Impact Twin & Smart Alerts",
        "📊 Model Validation & Feature Insights"
    ])

    # =========================================================================
    # TAB 1: REAL-TIME SIMULATOR (Preserved)
    # =========================================================================
    with tab_sim:
        predicted_demand = predict_demand(model, feature_names, inputs)
        decision = generate_recommendation(
            forecast_demand=predicted_demand,
            hour=hour,
            workingday=workingday,
            weathersit=weather_sit,
            temp_celsius=temp_c
        )

        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Predicted Demand",
                value=f"{int(round(predicted_demand))} bikes/hr",
                delta=f"{int(round(predicted_demand - 190))} vs Avg (190)"
            )
        with col2:
            risk_class = f"risk-{decision['risk_level'].lower()}"
            st.markdown(f"""
            <div style="padding-top: 5px;">
                <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 4px;">OPERATIONAL RISK</div>
                <span class="risk-badge {risk_class}" style="font-size: 1.15rem;">
                    ● {decision['risk_level'].upper()}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.metric(
                label="Rebalancing Vans Needed",
                value=f"{decision['trucks_recommended']} Vans",
                delta="Active Shuttle" if decision['trucks_recommended'] >= 2 else "Standby"
            )
        with col4:
            st.metric(
                label="Field Technicians Needed",
                value=f"{decision['staffing_estimate']} Staff",
                delta="Station & Logistics"
            )

        # Action Directive Banner
        alert_color = {"Low": "#E0F2FE", "Moderate": "#FEF3C7", "High": "#FFEDD5", "Critical": "#FEE2E2"}.get(decision["risk_level"], "#F1F5F9")
        border_color = {"Low": "#0284C7", "Moderate": "#D97706", "High": "#EA580C", "Critical": "#DC2626"}.get(decision["risk_level"], "#94A3B8")

        st.markdown(f"""
        <div style="background-color: {alert_color}; border-left: 5px solid {border_color}; padding: 14px 18px; border-radius: 6px; margin: 15px 0;">
            <strong style="color: {border_color}; font-size: 1.05rem;">📋 Operator Action Directive:</strong><br>
            <span style="font-size: 1.0rem; color: #1E293B;">{decision['recommended_action']}</span><br>
            <small style="color: #475569; margin-top: 6px; display: inline-block;"><strong>Reasoning:</strong> {decision['explanation']}</small>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("24-Hour Forecast Curve Under Current Weather")
        hours_list = list(range(24))
        day_preds = []
        for h in hours_list:
            sim_inputs = inputs.copy()
            sim_inputs["hr"] = h
            day_preds.append(predict_demand(model, feature_names, sim_inputs))

        df_day = pd.DataFrame({"Hour": hours_list, "Predicted_Demand": day_preds})
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=100, fillcolor="#22C55E", opacity=0.08, line_width=0, annotation_text="Low Risk (<100)")
        fig.add_hrect(y0=100, y1=300, fillcolor="#EAB308", opacity=0.08, line_width=0, annotation_text="Moderate (100-300)")
        fig.add_hrect(y0=300, y1=550, fillcolor="#F97316", opacity=0.08, line_width=0, annotation_text="High Risk (300-550)")
        fig.add_hrect(y0=550, y1=max(800, max(day_preds) + 50), fillcolor="#EF4444", opacity=0.08, line_width=0, annotation_text="Critical (>550)")

        fig.add_trace(go.Scatter(
            x=df_day["Hour"], y=df_day["Predicted_Demand"],
            mode="lines+markers", name="Forecast Demand",
            line=dict(color="#2563EB", width=3), marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=[hour], y=[predicted_demand],
            mode="markers", name=f"Selected Hour ({hour}:00)",
            marker=dict(color="#DC2626", size=14, symbol="diamond", line=dict(color="white", width=2))
        ))
        fig.update_layout(
            xaxis_title="Hour of Day (00:00 to 23:00)",
            yaxis_title="Total Bike Rentals / Hour",
            hovermode="x unified",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            height=420, margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # TAB 2: CUSTOM PREDICTION & BULK FORECASTING (Dynamic to Active Dataset)
    # =========================================================================
    with tab_bulk:
        st.header("📥 Custom Prediction & Bulk Forecasting")
        st.markdown(
            "Predict bike demand on arbitrary manual scenarios or run high-performance batch forecasts "
            "directly on the **Currently Active Dataset**."
        )

        # Active Dataset Overview Banner
        st.markdown(f"""
        <div class="dataset-badge">
            <span style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">📁 Active Dataset: {active_profile['name']}</span><br>
            <span style="color: #475569;">
                <strong>Total Records:</strong> {active_profile['total_rows']:,} | 
                <strong>Valid Records:</strong> {active_profile['valid_rows']:,} | 
                <strong>Quarantined:</strong> {active_profile['invalid_rows']} | 
                <strong>Schema:</strong> <code>{active_profile['schema']}</code> | 
                <strong>Target:</strong> <code>{target_badge}</code>
            </span>
        </div>
        """, unsafe_allow_html=True)

        bulk_mode = st.radio("Select Prediction Mode:", ["A. Manual Single-Record Prediction", "B. Batch Forecast on Active Dataset"], horizontal=True)

        if bulk_mode == "A. Manual Single-Record Prediction":
            st.subheader("Manual Input: Predict with New Data")
            st.caption("Enter arbitrary operational conditions to test the trained model on new unobserved scenarios.")

            with st.form("manual_predict_form"):
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    m_hour = st.number_input("Hour of Day (0–23):", min_value=0, max_value=23, value=hour)
                    m_temp = st.number_input("Temperature (°C):", min_value=-20.0, max_value=50.0, value=float(temp_c), step=0.5)
                    m_humidity = st.number_input("Relative Humidity (%):", min_value=0.0, max_value=100.0, value=float(humidity), step=1.0)
                with f_col2:
                    m_wind = st.number_input("Wind Speed (km/h):", min_value=0.0, max_value=80.0, value=float(windspeed), step=1.0)
                    m_weather = st.selectbox(
                        "Weather Condition:",
                        options=[1, 2, 3, 4],
                        index=weather_sit - 1,
                        format_func=lambda x: f"{x}: {weather_labels[x]}"
                    )
                    m_season = st.selectbox("Season:", options=[1, 2, 3, 4], index=season - 1, format_func=lambda x: season_labels[x])
                with f_col3:
                    m_wkday = st.selectbox("Day of Week:", options=[0, 1, 2, 3, 4, 5, 6], index=weekday, format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
                    m_is_work = st.selectbox("Working Day:", options=[1, 0], index=0 if workingday == 1 else 1, format_func=lambda x: "Yes (Weekday)" if x == 1 else "No (Weekend/Holiday)")
                    m_month = st.number_input("Month (1–12):", min_value=1, max_value=12, value=month)

                submitted = st.form_submit_button("🚀 Predict Demand & Generate Operations Plan", use_container_width=True)

            if submitted:
                m_inputs = {
                    "hr": m_hour, "workingday": m_is_work, "weekday": m_wkday, "mnth": m_month,
                    "season": m_season, "holiday": 0, "weathersit": m_weather,
                    "temp_celsius": m_temp, "humidity_pct": m_humidity, "windspeed_kmh": m_wind
                }
                m_pred = predict_demand(model, feature_names, m_inputs)
                m_rec = generate_recommendation(m_pred, hour=m_hour, workingday=m_is_work, weathersit=m_weather, temp_celsius=m_temp)

                st.markdown("---")
                st.info("ℹ️ **Notice for Evaluators**: The results below represent an **AI Model Forecast** generated by our Random Forest pipeline, not historical ride counts.")

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Predicted Demand", f"{int(round(m_pred))} bikes/hr")
                with mc2:
                    st.markdown(f"""
                    <div style="padding-top: 5px;">
                        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 4px;">OPERATIONAL RISK</div>
                        <span class="risk-badge risk-{m_rec['risk_level'].lower()}" style="font-size: 1.15rem;">● {m_rec['risk_level'].upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with mc3:
                    st.metric("Rebalancing Vans", f"{m_rec['trucks_recommended']} Vans")
                with mc4:
                    st.metric("Field Technicians", f"{m_rec['staffing_estimate']} Staff")

                st.success(f"**Directive**: {m_rec['recommended_action']}\n\n**Reasoning**: {m_rec['explanation']}")

        else:
            st.subheader(f"Batch Forecasting on Active Dataset ({active_profile['name']})")

            # Check model options
            has_client_model = active_profile["id"] in dm.client_models
            model_choices = []
            if active_profile["is_compatible"]:
                model_choices.append("Pretrained CityCycle Model (Random Forest, R²=0.9012)")
            if has_client_model:
                c_mod = dm.client_models[active_profile["id"]]
                model_choices.append(f"Client-Specific Model ({c_mod['model_type']}, Test MAE: {c_mod['metrics']['mae']})")

            col_model_sel, col_train_btn = st.columns([2, 1])

            with col_model_sel:
                if model_choices:
                    selected_model_mode = st.selectbox("Choose Forecasting Model:", options=model_choices)
                else:
                    selected_model_mode = None
                    st.warning("⚠️ The pretrained model is incompatible with this custom dataset. Please train a client-specific model below.")

            with col_train_btn:
                if active_profile["has_ground_truth"]:
                    st.write("")
                    if st.button("🛠️ Train Client Model on this Dataset", help="Trains an isolated model on the active dataset using chronological 80/20 splitting."):
                        with st.spinner("Training client-specific model..."):
                            try:
                                client_pkg = train_client_model(active_profile["valid_df"], target_col=active_profile["target_col"])
                                dm.client_models[active_profile["id"]] = client_pkg
                                st.success(f"Trained Client Model! Test MAE: {client_pkg['metrics']['mae']} bikes/hr (Baseline: {client_pkg['metrics']['baseline_mae']})")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Training failed: {e}")

            # Sample Template Download helper
            with st.expander("📄 Download Sample Templates & Column Guide"):
                template_df = generate_sample_template()
                csv_buffer = io.StringIO()
                template_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📄 Download Sample CSV Template",
                    data=csv_buffer.getvalue().encode("utf-8"),
                    file_name="citycycle_sample_template.csv",
                    mime="text/csv"
                )

            # Execution Button
            if st.button("🚀 Generate Forecasts on Active Dataset", type="primary", use_container_width=True):
                if active_profile["valid_df"].empty:
                    st.error("Cannot forecast: Active dataset has no valid records.")
                elif not active_profile["is_compatible"] and not has_client_model:
                    st.error(f"Cannot forecast: Dataset missing required features ({', '.join(active_profile['missing_features'])}). Train a client model first.")
                else:
                    # Choose model
                    if has_client_model and "Client-Specific" in selected_model_mode:
                        active_model = dm.client_models[active_profile["id"]]["model"]
                        active_features = dm.client_models[active_profile["id"]]["feature_names"]
                    else:
                        active_model = model
                        active_features = feature_names

                    with st.spinner(f"Scoring {active_profile['valid_rows']:,} records..."):
                        batch_results = predict_batch(active_model, active_features, active_profile["valid_df"])
                        st.session_state["active_batch_results"] = batch_results

            # Render batch results if computed for the active dataset
            if "active_batch_results" in st.session_state and not st.session_state["active_batch_results"].empty:
                batch_results_df = st.session_state["active_batch_results"]
                has_actual_cnt = "cnt" in batch_results_df.columns and batch_results_df["cnt"].notnull().any()

                if has_actual_cnt:
                    upload_mae = batch_results_df["abs_error"].mean()
                    upload_rmse = np.sqrt((batch_results_df["prediction_error"] ** 2).mean())
                    st.info(
                        f"🎯 **Ground-Truth Target ('cnt') Detected**: Successfully evaluated model against actual rentals! "
                        f"**Upload MAE:** `{upload_mae:.2f} bikes/hr` | **Upload RMSE:** `{upload_rmse:.2f} bikes/hr`."
                    )

                st.success(f"✅ Displaying forecasts for **{len(batch_results_df):,} records**.")

                # Batch Summary KPIs
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                with b_col1:
                    st.metric("Total Records", f"{len(batch_results_df):,}")
                with b_col2:
                    st.metric("Avg Predicted Demand", f"{batch_results_df['predicted_demand'].mean():.1f} bikes/hr")
                with b_col3:
                    st.metric("Peak Forecast", f"{batch_results_df['predicted_demand'].max()} bikes/hr")
                with b_col4:
                    critical_count = (batch_results_df["risk_level"] == "Critical").sum()
                    st.metric("Critical Surge Hours", f"{critical_count} Hours")

                # Interactive Timeline / Comparison Chart
                if has_actual_cnt:
                    st.subheader("Comparative Trajectory: Actual ('cnt') vs. Predicted Demand")
                    plot_sample = batch_results_df.head(168) if len(batch_results_df) > 168 else batch_results_df
                    fig_batch = go.Figure()
                    fig_batch.add_trace(go.Scatter(
                        x=list(range(len(plot_sample))),
                        y=plot_sample["cnt"],
                        mode="lines",
                        name="Actual Rentals (cnt)",
                        line=dict(color="#10B981", width=2)
                    ))
                    fig_batch.add_trace(go.Scatter(
                        x=list(range(len(plot_sample))),
                        y=plot_sample["predicted_demand"],
                        mode="lines",
                        name="Predicted Demand",
                        line=dict(color="#2563EB", width=2, dash="dot")
                    ))
                    sample_label = f" (Displaying first {len(plot_sample)} hours)" if len(batch_results_df) > 168 else ""
                    fig_batch.update_layout(
                        title=f"Actual vs. Predicted Hourly Demand{sample_label}",
                        xaxis_title="Record Index (Hourly Sequence)",
                        yaxis_title="Bikes / Hour",
                        hovermode="x unified",
                        height=380,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_batch, use_container_width=True)
                else:
                    fig_batch = px.bar(
                        batch_results_df.head(100),
                        x="record_id",
                        y="predicted_demand",
                        color="risk_level",
                        color_discrete_map={"Low": "#22C55E", "Moderate": "#EAB308", "High": "#F97316", "Critical": "#EF4444"},
                        title="Batch Forecast Demand Profile by Record ID (First 100 records)",
                        labels={"record_id": "Record Identifier", "predicted_demand": "Predicted Demand (bikes/hr)", "risk_level": "Risk Level"}
                    )
                    fig_batch.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40))
                    st.plotly_chart(fig_batch, use_container_width=True)

                # Results Table Preview
                st.subheader("Preview Predictions & Operations Table")
                st.dataframe(batch_results_df, use_container_width=True)

                # Download button
                out_buffer = io.StringIO()
                batch_results_df.to_csv(out_buffer, index=False)
                out_bytes = out_buffer.getvalue().encode("utf-8")

                st.download_button(
                    label=f"💾 Download Predictions as CSV ({active_profile['name'][:25]})",
                    data=out_bytes,
                    file_name=f"citycycle_predictions_{active_profile['id']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # =========================================================================
    # TAB 3: WEATHER IMPACT TWIN & SMART ALERTS (Feature 2)
    # =========================================================================
    with tab_twin:
        st.header("🌦️ Weather Impact Twin & Smart Weather Alerts")
        st.markdown(
            "Compare your baseline operations against simulated adverse weather scenarios to quantify demand sensitivity "
            "and receive actionable mitigation alerts."
        )

        col_twin_cfg1, col_twin_cfg2 = st.columns([1, 1])

        with col_twin_cfg1:
            st.subheader("1. Select Adverse Weather Twin Scenario")
            twin_preset_name = st.selectbox(
                "Simulated Weather Twin Preset:",
                list(TWIN_PRESETS.keys()) + ["⚙️ Custom Weather Twin"]
            )

            if twin_preset_name in TWIN_PRESETS:
                p_cfg = TWIN_PRESETS[twin_preset_name]
                st.caption(f"**Preset Description**: {p_cfg['description']}")
                twin_weather = p_cfg["weathersit"]
                twin_temp = temp_c + p_cfg.get("temp_offset", 0.0) if "temp_override" not in p_cfg else p_cfg["temp_override"]
                twin_humidity = p_cfg["humidity_pct"]
                twin_wind = p_cfg["windspeed_kmh"]
            else:
                twin_weather = st.selectbox("Twin Weather Condition:", options=[1, 2, 3, 4], index=2, format_func=lambda x: f"{x}: {weather_labels[x]}")
                twin_temp = st.slider("Twin Temperature (°C):", -10.0, 45.0, 10.0, step=0.5)
                twin_humidity = st.slider("Twin Humidity (%):", 0, 100, 90)
                twin_wind = st.slider("Twin Wind Speed (km/h):", 0.0, 70.0, 30.0, step=1.0)

        with col_twin_cfg2:
            st.subheader("2. Scenario Comparison Setup")
            st.markdown(f"""
            * **Baseline Weather**: {weather_labels[weather_sit]} | **{temp_c:.1f}°C** | Humidity: **{humidity}%** | Wind: **{windspeed:.1f} km/h**
            * **Twin Weather**: {weather_labels[twin_weather]} | **{twin_temp:.1f}°C** | Humidity: **{twin_humidity}%** | Wind: **{twin_wind:.1f} km/h**
            * **Evaluation Time**: **{hour:02d}:00** on a **{'Working Day' if workingday==1 else 'Weekend/Holiday'}**
            """)

        # Assemble Twin inputs
        twin_inputs = inputs.copy()
        twin_inputs["weathersit"] = twin_weather
        twin_inputs["temp_celsius"] = twin_temp
        twin_inputs["humidity_pct"] = twin_humidity
        twin_inputs["windspeed_kmh"] = twin_wind

        # Execute Twin Comparison
        twin_result = simulate_weather_twin_single(
            model=model,
            feature_names=feature_names,
            base_inputs=inputs,
            twin_inputs=twin_inputs,
            scenario_name=twin_preset_name
        )

        st.markdown("---")
        st.subheader("Side-by-Side Impact Analysis")

        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric(
                label="Baseline Demand",
                value=f"{twin_result['baseline_demand']} bikes/hr",
                help="Predicted demand under standard baseline conditions"
            )
        with tc2:
            st.metric(
                label="Twin Forecast Demand",
                value=f"{twin_result['twin_demand']} bikes/hr",
                delta=f"{twin_result['delta_demand']:+d} ({twin_result['pct_change']:+.1f}%)"
            )
        with tc3:
            st.metric(
                label="Rebalancing Vans Needed",
                value=f"{twin_result['twin_vans']} Vans",
                delta=f"{twin_result['twin_vans'] - twin_result['baseline_vans']:+d} Vans vs Base"
            )
        with tc4:
            st.metric(
                label="Field Staff Allocation",
                value=f"{twin_result['twin_staff']} Staff",
                delta=f"{twin_result['twin_staff'] - twin_result['baseline_staff']:+d} Staff vs Base"
            )

        # Smart Weather Alerts Section
        st.subheader("🚨 Smart Weather Operator Alerts")
        st.caption("Contextual decision alerts generated from simulated weather variances:")

        for al in twin_result["alerts"]:
            css_box = f"alert-{al['level']}"
            badge_icon = "⚠️" if al["level"] == "warning" else ("🚨" if al["level"] == "critical" else "ℹ️")
            st.markdown(f"""
            <div class="alert-box {css_box}">
                <div style="font-weight: 700; font-size: 1.05rem;">{badge_icon} [{al['type'].upper()}]: {al['title']}</div>
                <div style="margin-top: 4px; color: #1E293B;">{al['message']}</div>
                <small style="color: #64748B;">*Notice: Simulated Scenario Alert based on user-modeled weather parameters.*</small>
            </div>
            """, unsafe_allow_html=True)

        # 24-Hour Comparative Curve
        st.subheader("24-Hour Trajectory: Baseline vs. Weather Twin")
        twin_df_24 = simulate_weather_twin_24h(model, feature_names, inputs, twin_inputs)

        fig_twin_curve = go.Figure()

        # Risk zone background
        fig_twin_curve.add_hrect(y0=0, y1=100, fillcolor="#22C55E", opacity=0.06, line_width=0, annotation_text="Low (<100)")
        fig_twin_curve.add_hrect(y0=100, y1=300, fillcolor="#EAB308", opacity=0.06, line_width=0, annotation_text="Moderate (100-300)")
        fig_twin_curve.add_hrect(y0=300, y1=550, fillcolor="#F97316", opacity=0.06, line_width=0, annotation_text="High (300-550)")
        fig_twin_curve.add_hrect(y0=550, y1=max(800, max(twin_df_24["Baseline_Demand"].max(), twin_df_24["Twin_Demand"].max()) + 60), fillcolor="#EF4444", opacity=0.06, line_width=0, annotation_text="Critical (>550)")

        # Baseline Curve
        fig_twin_curve.add_trace(go.Scatter(
            x=twin_df_24["Hour"],
            y=twin_df_24["Baseline_Demand"],
            mode="lines+markers",
            name="Normal Weather Baseline",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=5)
        ))

        # Twin Curve
        fig_twin_curve.add_trace(go.Scatter(
            x=twin_df_24["Hour"],
            y=twin_df_24["Twin_Demand"],
            mode="lines+markers",
            name=f"Twin: {twin_preset_name.split()[1] if ' ' in twin_preset_name else twin_preset_name}",
            line=dict(color="#D97706", width=3, dash="dot"),
            marker=dict(size=5)
        ))

        fig_twin_curve.update_layout(
            xaxis_title="Hour of Day (00:00 to 23:00)",
            yaxis_title="Total Hourly Rentals (cnt)",
            hovermode="x unified",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            height=420,
            margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig_twin_curve, use_container_width=True)

    # =========================================================================
    # TAB 4: MODEL VALIDATION & FEATURE INSIGHTS
    # =========================================================================
    with tab_valid:
        st.header("📊 Model Validation on Unseen Test Period (3,476 Hours)")
        col_m1, col_m2, col_m3 = st.columns(3)

        rf_metrics = metrics_data.get("metrics", {}).get("random_forest", {})
        base_metrics = metrics_data.get("metrics", {}).get("baseline", {})

        with col_m1:
            st.metric(
                label="R² Score (Variance Explained)",
                value=f"{rf_metrics.get('r2', 0.9012):.4f}",
                delta=f"+{rf_metrics.get('r2', 0.9012) - base_metrics.get('r2', 0.6725):.4f} vs Baseline"
            )
        with col_m2:
            st.metric(
                label="MAE (Mean Absolute Error)",
                value=f"{rf_metrics.get('mae', 44.76):.2f} bikes",
                delta=f"-{base_metrics.get('mae', 93.68) - rf_metrics.get('mae', 44.76):.2f} bikes error",
                delta_color="inverse"
            )
        with col_m3:
            st.metric(
                label="RMSE (Root Mean Sq Error)",
                value=f"{rf_metrics.get('rmse', 69.31):.2f} bikes",
                delta=f"-{base_metrics.get('rmse', 126.17) - rf_metrics.get('rmse', 69.31):.2f} bikes",
                delta_color="inverse"
            )

        st.info("""
        **Validation & Methodology Integrity**:
        - **Chronological Split**: Trained on the earlier 80% of timestamps (13,903 hours), tested on the latest 20% (3,476 hours) to prevent future-data contamination.
        - **Strict Leakage Prevention**: `casual` and `registered` columns were strictly quarantined from input features so the model cannot cheat by summing rider subcategories.
        - **Non-Linear Advantage**: Random Forest ensemble captures compound conditional commute spikes, reducing MAE by 52.2% compared to the linear baseline.
        """)

        st.subheader("Top Predictive Features in Demand Model")
        top_features = metrics_data.get("top_features", {})
        if top_features:
            feat_df = pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"]).head(8)
            feat_df = feat_df.sort_values(by="Importance", ascending=True)

            fig_bar = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                title="Random Forest Feature Importances"
            )
            fig_bar.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
