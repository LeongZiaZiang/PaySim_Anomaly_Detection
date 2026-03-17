import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# API base URL
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Real-Time Fraud Detection Dashboard")
st.markdown("**PaySim Transaction Monitoring — XGBoost vs River HalfSpaceTrees**")

# Auto refresh every 3 seconds
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 3)
st.sidebar.markdown("---")
st.sidebar.markdown("### Models")
st.sidebar.markdown("✅ **XGBoost** — Pre-trained batch model")
st.sidebar.markdown("✅ **River** — Online learning model")

placeholder = st.empty()

while True:
    try:
        # Fetch data from API
        summary = requests.get(f"{API_URL}/summary").json()
        xgb_stats = requests.get(f"{API_URL}/xgb/stats").json()
        river_stats = requests.get(f"{API_URL}/river/stats").json()
        xgb_alerts = requests.get(f"{API_URL}/xgb/alerts?limit=20").json()
        river_alerts = requests.get(f"{API_URL}/river/alerts?limit=20").json()

        with placeholder.container():

            # --- Row 1: Total Fraud ---
            st.subheader("📊 Total Detection")
            col1, col2 = st.columns(2)

            col1.metric(
                    "XGB Total Fraud Detected",
                    f"{summary['xgboost']['total_fraud']:,}"
                )
            col2.metric(
                    "River Total Anomaly Detected",
                    f"{summary['river']['total_anomaly']:,}"
                )

            st.markdown("---")

            # --- Row 2: Summary metrics ---
            st.subheader("📊 Live Summary")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "XGB Transactions Processed",
                f"{summary['xgboost']['total_processed']:,}"
            )
            col2.metric(
                "XGB Fraud Detected",
                f"{summary['xgboost']['fraud_detected']:,}",
                f"{summary['xgboost']['fraud_rate']}%"
            )
            col3.metric(
                "River Transactions Processed",
                f"{summary['river']['total_processed']:,}"
            )
            col4.metric(
                "River Anomalies Detected",
                f"{summary['river']['anomalies_detected']:,}",
                f"{summary['river']['anomaly_rate']}%"
            )

            st.markdown("---")

            # --- Row 3: Model performance ---
            st.subheader("🎯 Model Performance")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### XGBoost")
                if 'accuracy' in xgb_stats:
                    st.metric("Accuracy", f"{xgb_stats['accuracy']:.2f}%")
                    xgb_col1, xgb_col2, xgb_col3 = st.columns(3)
                    xgb_col1.metric("✅ Fraud Caught", xgb_stats['fraud_caught'])
                    xgb_col2.metric("❌ Fraud Missed", xgb_stats['fraud_missed'])
                    xgb_col3.metric("⚠️ False Alarms", xgb_stats['false_alarms'])
                else:
                    st.info("Waiting for data...")

            with col2:
                st.markdown("#### River HalfSpaceTrees")
                if 'accuracy' in river_stats:
                    st.metric("Accuracy", f"{river_stats['accuracy']:.2f}%")
                    rv_col1, rv_col2, rv_col3 = st.columns(3)
                    rv_col1.metric("✅ Fraud Caught", river_stats['fraud_caught'])
                    rv_col2.metric("❌ Fraud Missed", river_stats['fraud_missed'])
                    rv_col3.metric("⚠️ False Alarms", river_stats['false_alarms'])
                else:
                    st.info("Waiting for data...")

            st.markdown("---")

            # --- Row 4: Latest alerts ---
            st.subheader("🚨 Latest Fraud Alerts")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### XGBoost Alerts")
                if xgb_alerts['latest_alerts']:
                    df_xgb = pd.DataFrame(xgb_alerts['latest_alerts'])
                    df_xgb = df_xgb[['transaction_id', 'amount', 
                                      'fraud_probability', 'actual']]
                    df_xgb['amount'] = df_xgb['amount'].apply(
                        lambda x: f"${x:,.2f}")
                    df_xgb['correct'] = df_xgb.apply(
                        lambda r: '✅' if r['actual'] == 1 else '⚠️', axis=1)
                    st.dataframe(df_xgb, use_container_width=True)
                else:
                    st.info("No fraud alerts yet...")

            with col2:
                st.markdown("#### River Alerts")
                if river_alerts['latest_alerts']:
                    df_river = pd.DataFrame(river_alerts['latest_alerts'])
                    df_river = df_river[['transaction_id', 'amount',
                                         'anomaly_score', 'actual']]
                    df_river['amount'] = df_river['amount'].apply(
                        lambda x: f"${x:,.2f}")
                    df_river['correct'] = df_river.apply(
                        lambda r: '✅' if r['actual'] == 1 else '⚠️', axis=1)
                    st.dataframe(df_river, use_container_width=True)
                else:
                    st.info("No anomaly alerts yet...")

            st.markdown("---")
            st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
                      f"Auto-refreshing every {refresh_rate}s")

    except Exception as e:
        st.error(f"API not reachable: {e}. Make sure api.py is running.")

    time.sleep(refresh_rate)