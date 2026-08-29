import streamlit as st
import requests


st.set_page_config(
    page_title="Predictive Maintenance Platform",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Predictive Maintenance Platform")
st.write("AI-powered machine failure risk assessment using Gradient Boosting.")
st.divider()

st.subheader("Machine Measurements")
st.write("Enter the current operating conditions of the machine.")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temperature = st.number_input(
        "Air Temperature [K]",
        250.0,
        350.0,
        298.1,
        0.1
    )
    process_temperature = st.number_input(
        "Process Temperature [K]",
        250.0,
        350.0,
        308.6,
        0.1
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        0.0,
        5000.0,
        1551.0,
        1.0
    )

    torque = st.number_input(
        "Torque [Nm]",
        0.0,
        100.0,
        42.8,
        0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        0.0,
        300.0,
        0.0,
        1.0
    )

st.divider()

if st.button(
    "Run Failure Prediction",
    type="primary",
    use_container_width=True
):
    payload = {
        "machine_type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    try:
        response = requests.post(
            "http://api:8000/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probability = result["failure_probability"]
            feature_importance = result.get(
                "feature_importance",
                {}
            )
            health = (1 - probability) * 100

            if probability < 0.20:
                status = "Healthy"
                status_color = "#22c55e"
                recommendation = (
                    "Continue normal operation. "
                    "No immediate maintenance action is indicated."
                )
                recommendation_icon = "🟢"

            elif probability < 0.35:
                status = "Warning"
                status_color = "#f59e0b"
                recommendation = (
                    "Schedule an inspection or maintenance check. "
                    "The machine shows an elevated failure risk."
                )
                recommendation_icon = "🟡"

            else:
                status = "Critical"
                status_color = "#ef4444"
                recommendation = (
                    "Inspect the machine before continued operation. "
                    "A high failure risk has been detected."
                )
                recommendation_icon = "🔴"

            st.subheader("Prediction Result")
            st.markdown(
                f"""<div style="background:#111827;border:1px solid #263244;border-radius:16px;padding:28px;text-align:center;margin:18px 0;">

<div style="color:#9ca3af;font-size:16px;font-weight:600;letter-spacing:0.5px;margin-bottom:18px;">
ESTIMATED MACHINE HEALTH
</div>

<div style="width:190px;height:190px;border-radius:50%;background:conic-gradient({status_color} {health}%,#1f2937 {health}% 100%);margin:auto;display:flex;align-items:center;justify-content:center;">

<div style="width:142px;height:142px;border-radius:50%;background:#111827;display:flex;flex-direction:column;align-items:center;justify-content:center;">

<div style="font-size:34px;font-weight:700;color:#f9fafb;">
{health:.2f}%
</div>

<div style="font-size:15px;color:{status_color};font-weight:600;margin-top:4px;">
{status}
</div>

</div>
</div>

<p style="color:#9ca3af;margin-top:18px;margin-bottom:0;">
Estimated from the model's predicted failure probability
</p>

</div>""",
                unsafe_allow_html=True,
            )

            if prediction == 1:
                st.error(
                    "⚠️ Machine failure predicted"
                )

            else:
                st.success(
                    "✓ No machine failure predicted"
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Failure Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Prediction",
                    "Failure"
                    if prediction == 1
                    else "No Failure"
                )

            with col3:
                st.metric(
                    "Risk Level",
                    status
                )

            st.divider()
            st.subheader("Operating Conditions")
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.caption("Machine")
                st.write(f"### {machine_type}")

            with col2:
                st.caption("Air Temp.")
                st.write(
                    f"### {air_temperature:.1f} K"
                )

            with col3:
                st.caption("Process Temp.")
                st.write(
                    f"### {process_temperature:.1f} K"
                )

            with col4:
                st.caption("Speed")
                st.write(
                    f"### {rotational_speed:.0f} rpm"
                )

            with col5:
                st.caption("Torque")
                st.write(
                    f"### {torque:.1f} Nm"
                )

            with col6:
                st.caption("Tool Wear")
                st.write(
                    f"### {tool_wear:.0f} min"
                )

            st.divider()
            st.subheader("Recommended Action")

            st.markdown(
                f"""<div style="background:#111827;border:1px solid {status_color};border-left:5px solid {status_color};border-radius:12px;padding:18px 22px;margin:10px 0 20px 0;">

<div style="font-size:18px;font-weight:650;color:#f9fafb;margin-bottom:8px;">
{recommendation_icon} {status} Risk
</div>

<div style="color:#d1d5db;font-size:15px;line-height:1.5;">
{recommendation}
</div>

</div>""",
                unsafe_allow_html=True,
            )

            st.caption(
                "Recommendation is generated from predefined risk thresholds "
                "applied to the model's predicted failure probability."
            )
            st.divider()

            st.subheader("Why This Prediction?")
            st.write(
                "The model identifies which machine measurements contribute "
                "most strongly to its prediction."
            )

            if feature_importance:
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                top_features = sorted_features[:3]
                other_features = sorted_features[3:]
                st.markdown(
                    "**Top Model Drivers**"
                )

                for feature, importance in top_features:
                    percentage = importance * 100
                    st.markdown(
                        f"**{feature}** — **{percentage:.1f}%**"
                    )
                    st.progress(
                        min(
                            100,
                            int(percentage)
                        )
                    )

                if other_features:
                    with st.expander(
                        "Other model features"
                    ):
                        for feature, importance in other_features:
                            st.write(
                                f"{feature}: "
                                f"**{importance * 100:.1f}%**"
                            )

                if len(top_features) >= 2:
                    st.info(
                        f"**Model insight:** "
                        f"{top_features[0][0]} and "
                        f"{top_features[1][0]} are the strongest "
                        f"model drivers for this prediction."
                    )
            else:
                st.info(
                    "Feature importance information is not available "
                    "from the prediction API."
                )

            st.divider()
            st.subheader("Model Information")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("**Algorithm**")
                st.write("Gradient Boosting")

            with col2:
                st.markdown("**Task**")
                st.write("Binary Classification")

            with col3:
                st.markdown("**Target**")
                st.write("Machine Failure")

            with col4:
                st.markdown("**Dataset**")
                st.write("AI4I 2020")

            st.divider()
            st.caption(
                "Predictive Maintenance Platform • Built with Python, "
                "Scikit-learn, FastAPI and Streamlit"
            )

        else:
            st.error(
                f"API request failed with status code "
                f"{response.status_code}"
            )

    except requests.exceptions.RequestException:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure the FastAPI server is running."
        )