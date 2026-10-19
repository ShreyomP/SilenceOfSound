import os
import sys
import time
import random
import asyncio
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import AIS_API_KEY, DETECTION_THRESHOLD
from core.models import AcousticEngine
from core.fetcher import QuiltS3Fetcher
from core.maritime import MaritimeEngine
from core.analytics import AnalyticsEngine

HYDRO_LAT = 48.1357
HYDRO_LNG = -122.7597

def run_app():
    st.set_page_config(page_title="OrcaSafe Interface", layout="wide")
    st.title("OrcaSafe: Acoustic Mitigation Interface")
    st.markdown("---")

    if 'init' not in st.session_state:
        st.session_state.fetcher = QuiltS3Fetcher()
        model_path = PROJECT_ROOT / "models" / "orca_safe_brain.pth"
        st.session_state.acoustic = AcousticEngine(str(model_path) if model_path.exists() else "orca_safe_brain.pth")
        st.session_state.maritime = MaritimeEngine(AIS_API_KEY)
        st.session_state.analytics = AnalyticsEngine()
        st.session_state.init = True

    st.sidebar.subheader("System Control Panel")
    mode = st.sidebar.radio("Input Source", ["Live Stream", "Historical Baseline"])
    ship_mode = st.sidebar.radio("AIS Data Source", ["Live AIS Stream", "Tactical Dataset (Ever Summit)"])
    refresh_rate = st.sidebar.slider("Refresh Interval (s)", 5, 30, 15)

    # Acoustic analysis
    img, timestamp = None, None
    if mode == "Live Stream":
        result = st.session_state.fetcher.get_latest_spectrogram()
        if result[0] is not None:
            img, _, timestamp = result
        st.caption(f"Telemetry Synchronized: {timestamp}" if timestamp else "Synchronizing Signal...")
    else:
        category = st.sidebar.selectbox("Signal Profile", ["Orca", "Mixed", "Noise"])
        path = PROJECT_ROOT / "Data" / "InferenceData" / "MixedInference" / category
        if path.exists() and any(path.iterdir()):
            valid_files = [f for f in path.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            if valid_files:
                img = Image.open(random.choice(valid_files)).convert("RGB")
                st.caption(f"Baseline Source: {category}")

    if img:
        st.subheader("Acoustic Intelligence Analysis")
        col_img, col_metrics = st.columns([2, 1])
        col_img.image(img, use_container_width=True)

        prob = st.session_state.acoustic.run_inference(img)
        orca_detected = prob > DETECTION_THRESHOLD

        with col_metrics:
            st.metric("Detection Confidence", f"{prob:.2%}")
            if orca_detected:
                st.error("Orca detected")
            else:
                st.info("No orca detected")

        st.markdown("---")
        if orca_detected:
            st.subheader("Mitigation Analysis: Active Response Mode")

            if ship_mode == "Live AIS Stream":
                vessels_list = asyncio.run(
                    st.session_state.maritime.fetch_live_ais(node_lat=HYDRO_LAT, node_lng=HYDRO_LNG)
                )
                df = pd.DataFrame(vessels_list)
            else:
                df = pd.DataFrame([{
                    "name": "Ever Summit",
                    "sog": 17.2,
                    "latitude": 48.12,
                    "longitude": -122.75,
                    "status": "Underway (Engine)"
                }])

            if not df.empty and 'latitude' in df.columns:
                df['distance_km'] = df.apply(
                    lambda x: st.session_state.maritime.get_distance(HYDRO_LAT, HYDRO_LNG, x['latitude'], x['longitude']),
                    axis=1
                )
                phys = df.apply(lambda x: st.session_state.analytics.process_recovery(x['sog']), axis=1)
                df[['source_level_db', 'reclaimed_area_km2']] = pd.DataFrame(phys.tolist(), index=df.index)

                st.metric("Total Cumulative Habitat Recovery", f"{df['reclaimed_area_km2'].sum():.1f} km²")
                st.table(df[['name', 'status', 'sog', 'distance_km', 'source_level_db', 'reclaimed_area_km2']])

                # Vessels in red
                vessel_map_df = df[['latitude', 'longitude']].copy()
                vessel_map_df['color'] = '#FF0000'

                # Hydrophone in blue
                hydro_point = pd.DataFrame([{
                    'latitude': HYDRO_LAT,
                    'longitude': HYDRO_LNG,
                    'color': '#0000FF'
                }])

                final_map_df = pd.concat([vessel_map_df, hydro_point], ignore_index=True)
                st.map(final_map_df, color='color', size=20)
                st.info("Mitigation protocol active: Speed reduction advisory (7 knots) broadcast to vessels in range.")
            else:
                st.info("No commercial vessels detected in hydrophone range.")
        else:
            st.subheader("Mitigation Analysis: Standby Mode")
            st.caption("Maritime telemetry and AIS correlation are suspended until a positive detection is confirmed.")

    if mode == "Live Stream":
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    run_app()