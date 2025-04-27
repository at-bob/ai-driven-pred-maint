import streamlit as st
import pandas as pd
import os
from streamlit_autorefresh import st_autorefresh

# ====== Configuration ======
LOG_FILE = "/app/log/eep_data_log.csv"

# ====== Functions ======

def get_overall_status(risk_status, anomaly_status):
    if risk_status == "No Risk" and anomaly_status == "Normal":
        return "🟢 Healthy"
    elif risk_status == "Risk of Failure" and anomaly_status == "Anomaly Detected":
        return "🔴 Immediate Action Required"
    else:
        return "🟡 Attention Needed Soon"

def fetch_latest_dashboard_data():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            return df.iloc[-1]   # Get latest entry
    return None

# ====== Streamlit UI ======
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="centered")
st.title("🚨 Predictive Maintenance Dashboard")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="dashboard_refresh")

with st.spinner('Fetching latest system status...'):
    latest_data = fetch_latest_dashboard_data()

    if latest_data is not None:
        overall_status = get_overall_status(latest_data['risk_status'], latest_data['anomaly_status'])

        st.markdown(f"## Status: {overall_status}")
        st.write(f"**Risk Level:** {latest_data['risk_status']} ({float(latest_data['risk_probability'])*100:.1f}%)")
        st.write(f"**Anomaly Status:** {latest_data['anomaly_status']}")
        st.write(f"**Flagged Sensors:** {latest_data['flagged_sensors']}")
        st.write(f"**Last Update:** {latest_data['timestamp']}")
    else:
        st.warning("No data available yet. Waiting for input from the system...")

# ===== Footer =====
st.markdown("---")
st.caption("Powered by AI-Driven Predictive Maintenance System © 2025")
