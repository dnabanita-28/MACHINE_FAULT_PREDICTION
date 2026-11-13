import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Machine Health Monitor",
    page_icon="⚙️",
    layout="wide"
)

model = joblib.load("machine_fault_model.pkl")

st.title("⚙️ Industrial Machine Health Monitor")
st.caption("AI-Based Predictive Maintenance System")

st.divider()

st.subheader("Machine Operating Parameters")

c1, c2 = st.columns(2)

with c1:
    temp = st.slider("Temperature (°C)", 30, 110, 50)
    vibration = st.slider("Vibration (mm/s)", 1.0, 12.0, 3.0)

with c2:
    rpm = st.slider("Motor Speed (RPM)", 400, 3200, 1500)
    hours = st.slider("Operating Hours", 1, 24, 8)

if st.button("🔍 Predict Machine Condition", use_container_width=True):

    sample = pd.DataFrame([{
        "Temperature": temp,
        "Vibration": vibration,
        "RPM": rpm,
        "Operating_Hours": hours
    }])

    result = model.predict(sample)[0]

    st.divider()
    st.subheader("Machine Health Report")

    a, b, c = st.columns(3)

    a.metric("Temperature", f"{temp} °C")
    b.metric("Vibration", f"{vibration} mm/s")
    c.metric("Motor Speed", f"{rpm} RPM")

    if result == "Normal":
        st.success("NORMAL - Machine is operating normally.")
        st.write("Recommendation: Continue regular monitoring.")

    elif result == "Warning":
        st.warning("WARNING - Abnormal readings detected.")
        st.write("Recommendation: Inspect the machine and schedule maintenance.")

    else:
        st.error("FAULT DETECTED - Maintenance required!")
        st.write("Recommendation: Stop and inspect the machine before further operation.")

st.divider()
st.caption("Prototype using simulated sensor data and Random Forest ML.")
