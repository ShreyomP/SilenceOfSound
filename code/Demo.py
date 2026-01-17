import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import time

# --- CONFIGURATION & MOCK DATA ---
st.set_page_config(page_title="OrcaSafe: Real-Time Acoustic Mitigation", layout="wide")

def simulate_mitigation(initial_snr):
    """Calculates the 12dB slowdown recovery."""
    return initial_snr + 12.0

# --- HEADER ---
st.title("🐋 Silence in the Sound: Real-Time AI Mitigation")
st.markdown("### Southern Resident Killer Whale Protection System")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("Demo Controls")
mode = st.sidebar.radio("Simulation Scenario:", ["Quiet Water (+8dB)", "High Traffic (0dB Masking)", "Slowdown Active (+12dB Recovery)"])

# --- MAIN DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📡 Live Hydrophone Feed")
    # Placeholder for your actual spectrogram images
    # Replace 'orca_spec.png' with your actual project images
    st.image("https://via.placeholder.com/600x300.png?text=Spectrogram+Inference+Feed", use_container_width=True)
    
    snr_value = 8.0 if "Quiet" in mode else (0.0 if "High Traffic" in mode else 12.0)
    st.metric(label="Estimated SNR Proxy", value=f"{snr_value} dB", delta="12 dB (Recovery)" if "Recovery" in mode else None)

with col2:
    st.subheader("🤖 AI Detection Engine")
    confidence = 0.98 if "Quiet" in mode else (0.75 if "High Traffic" in mode else 0.99)
    st.progress(confidence)
    st.write(f"**Confidence Level:** {confidence:.1%}")
    
    if confidence > 0.85:
        st.success("✅ ORCA DETECTED")
    else:
        st.warning("⚠️ DETECTION MASKED BY NOISE")

st.divider()

# --- PHASE 3: SPATIAL RECOVERY VISUALIZATION ---
st.subheader("🗺️ Phase 3: Acoustic Habitat Recovery")
col3, col4 = st.columns([2, 1])

with col3:
    # Create a simple bubble map to show 'Acoustic Footprint'
    radius = 8.0 if "High Traffic" in mode else 2.0
    map_data = pd.DataFrame({'lat': [48.11], 'lon': [-122.76], 'radius': [radius]})
    
    fig = px.scatter_mapbox(map_data, lat="lat", lon="lon", size="radius", 
                            color_discrete_sequence=["red" if radius > 5 else "green"],
                            zoom=10, height=400)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("#### Mitigation Impact")
    impact_df = pd.DataFrame({
        "Metric": ["Noise Level", "Comm. Radius", "AI Reliability"],
        "Value": ["High" if radius > 5 else "Safe", 
                  f"{radius} km", 
                  "100%" if radius < 5 else "82%"]
    })
    st.table(impact_df)

if st.button("🚀 TRIGGER DYNAMIC SLOWDOWN"):
    with st.spinner('Calculating AIS Buffer and Notifying Vessels...'):
        time.sleep(2)
        st.balloons()
        st.success("Targeted MMSI Alert Sent: Speed Reduction to 7kts Confirmed.")