import streamlit as st
from streamlit_autorefresh import st_autorefresh  # Ensure streamlit-autorefresh is installed
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# URL of your Flask server
base_url = "http://127.0.0.1:5000"

st.title("Log Analyzer Dashboard")

# Function to fetch data from a given endpoint
def fetch_data(endpoint):
    response = requests.get(f"{base_url}{endpoint}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching data from {endpoint}: {response.status_code}")
        return {}

# Tabs for different data points
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Error Counts", "P Values", "Messages", "Timestamps", "Analyzed Messages"])

with tab1:
    st.header("AI Powered Log Analytics")
    live_error_counts_placeholder = st.empty()

with tab2:
    st.header("P Values")
    p_values_placeholder = st.empty()

with tab3:
    st.header("Messages")
    messages_placeholder = st.empty()

with tab4:
    st.header("Timestamps")
    timestamps_placeholder = st.empty()

with tab5:
    st.header("Analyzed Messages")
    analyzed_messages_placeholder = st.empty()

# Function to update data in the UI
def update_data():
    with st.spinner("Updating data..."):
        live_error_counts = fetch_data('/get_live_error_counts')
        p_values = fetch_data('/get_p')
        messages = fetch_data('/get_messages')
        timestamps = fetch_data('/get_timestamps')
        analyzed_messages = fetch_data('/get_analyzed_messages')

        # Update live error counts chart
        if live_error_counts:
            df = pd.DataFrame(live_error_counts.items(), columns=["Error Type", "Count"])
            
            # Generate a list of colors, one for each bar
            colors = px.colors.qualitative.Plotly * (len(df) // len(px.colors.qualitative.Plotly) + 1)
            colors = colors[:len(df)]

            fig = go.Figure(data=[go.Bar(
                x=df["Count"],
                y=df["Error Type"],
                orientation='h',
                marker=dict(color=colors)
            )])
            
            fig.update_layout(title="Live Error Counts")
            live_error_counts_placeholder.plotly_chart(fig)

        p_values_placeholder.json(p_values)
        messages_placeholder.json(messages)
        timestamps_placeholder.json(timestamps)
        analyzed_messages_placeholder.json(analyzed_messages)

        # Display messages with expander for details
        st.header("Errors with Details")
        for i, message in enumerate(messages):
            with st.expander(f"Message {message}", expanded=False):
                st.write(f"**Error Type:** {p_values[i]}")
                st.write(f"**Timestamp:** {timestamps[i]}")
                st.write(f"**Result:** {analyzed_messages[i]}")

# Auto-refresh every 10 seconds
refresh_interval = 10 * 1000  # Refresh interval in milliseconds

# Trigger auto-refresh
st_autorefresh(interval=refresh_interval, key="data_refresh")

# Update data on each refresh
update_data()
