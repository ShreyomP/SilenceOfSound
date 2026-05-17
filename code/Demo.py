import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import random
import asyncio
import websockets
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import librosa
import librosa.display
import io
import time
import subprocess
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from enum import Enum

# --- SYSTEM CONFIGURATION ---
FFMPEG_BINARY = "/opt/homebrew/bin/ffmpeg"
SHIP_TYPE_MAP = {
    70: "Cargo", 71: "Cargo (Haz A)", 80: "Tanker", 60: "Passenger", 0: "Unknown"
}

# --- ENUMS FOR STANDARDIZED OPERATIONS ---
class VesselSpeed(Enum):
    CRUISE_FAST = 22.0  # Fast passenger/container
    CRUISE_STD = 15.0   # Standard cargo
    ECHO_SLOW = 7.0     # ECHO Program target speed
    STATIONARY = 0.0    # At anchor/dock

# --- AIS NAVIGATION STATUS MAPPING ---
NAV_STATUS_MAP = {
    0: "Underway (Engine)",
    1: "At Anchor",
    2: "Not Under Command",
    3: "Restricted Maneuverability",
    4: "Constrained by Draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in Fishing",
    8: "Underway (Sailing)",
    15: "Undefined"
}

# --- MODULE 1: DATA ACQUISITION ---
class QuiltS3Fetcher:
    def __init__(self):
        self.bucket_name = "audio-orcasound-net"
        self.prefix = "rpi_port_townsend/hls/"
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def get_latest_spectrogram(self):
        try:
            # 1. Find the latest timestamp folder ("bucket") for the stream
            # Look back 30 days to avoid paginating through years of old folders
            lookback_folder = int(time.time()) - 30*24*60*60
            start_after_folder = f"{self.prefix}{lookback_folder}/"
            
            folder_response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix,
                Delimiter='/',
                StartAfter=start_after_folder
            )
            
            prefixes = folder_response.get('CommonPrefixes', [])
            if not prefixes:
                # Fallback if no recent data
                return None, None
                
            # The last prefix is the most recent timestamp folder
            latest_folder = prefixes[-1]['Prefix']

            # 2. Fetch the live.m3u8 playlist from the latest folder
            # This is significantly faster than paginating through thousands of .ts files
            playlist_key = f"{latest_folder}live.m3u8"
            
            playlist_obj = self.s3.get_object(Bucket=self.bucket_name, Key=playlist_key)
            playlist_content = playlist_obj['Body'].read().decode('utf-8')
            
            # 3. Parse the playlist to find the last .ts segment
            ts_files = [line.strip() for line in playlist_content.split('\n') if line.strip().endswith('.ts')]
            if not ts_files:
                return None, None
                
            latest_ts_filename = ts_files[-1]
            latest_ts_key = f"{latest_folder}{latest_ts_filename}"
            
            # Get metadata for the actual ts file
            ts_head = self.s3.head_object(Bucket=self.bucket_name, Key=latest_ts_key)
            
            # Ignore tiny/empty files (header only)
            if ts_head.get('ContentLength', 0) < 1000: 
                return None, None

            # 4. Download to memory instead of disk
            file_byte_string = self.s3.get_object(
                Bucket=self.bucket_name, 
                Key=latest_ts_key
            )['Body'].read()

            # 5. Use FFmpeg to convert TS to WAV in a memory pipe
            # This avoids the "live_capture.ts" disk write latency
            command = [
                'ffmpeg', '-i', 'pipe:0', 
                '-f', 'wav', '-ar', '22050', '-ac', '1', 
                'pipe:1', '-v', 'error'
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = process.communicate(input=file_byte_string)

            # 6. Load audio directly from the pipe output
            y, sr = librosa.load(io.BytesIO(out), sr=22050)
            
            # 7. Generate Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=10000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # 8. Render to Image buffer
            fig, ax = plt.subplots(figsize=(5, 5))
            plt.axis('off')
            librosa.display.specshow(S_dB, sr=sr, fmax=10000, ax=ax, cmap='magma')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            
            return Image.open(buf).convert("RGB"), ts_head['LastModified']

        except Exception as e:
            print(f"Error fetching latest spectrogram: {e}")
            return None, None
   
   # --- MODULE 2: INFERENCE ENGINE ---
class AcousticEngine:
    def __init__(self, model_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, path):
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())
        if os.path.exists(path): model.load_state_dict(torch.load(path, map_location=self.device))
        return model.to(self.device).eval()

    def run_inference(self, img):
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad(): return self.model(tensor).item()

# --- MODULE 3: MARITIME TELEMETRY ---
class MaritimeEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.bbox = [[48.0, -123.5], [48.5, -122.3]] 
        self.hydro_pos = (48.1357, -122.7597)

    async def fetch_live_ais(self):
        ships = {}
        try:
            async with websockets.connect("wss://stream.aisstream.io/v0/stream", open_timeout=10) as ws:
                await ws.send(json.dumps({"APIKey": self.api_key, "BoundingBoxes": [self.bbox]}))
                start_time = time.time()
                
                # Increased window to 10 seconds to better catch standard AIS broadcast intervals
                while time.time() - start_time < 10: 
                    try:
                        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                        
                        # Safety check: Ensure the message is actually a PositionReport
                        if "MetaData" in data and "Message" in data and "PositionReport" in data["Message"]:
                            mmsi = data["MetaData"]["MMSI"]
                            nav_code = data["Message"]["PositionReport"].get("NavigationalStatus", 15)
                            status_text = NAV_STATUS_MAP.get(nav_code, f"Status {nav_code}")
                            
                            ships[mmsi] = {
                                "Name": data["MetaData"].get("ShipName", "Unknown").strip(),
                                "latitude": data["MetaData"]["latitude"],
                                "longitude": data["MetaData"]["longitude"],
                                "SOG": data["Message"]["PositionReport"].get("Sog", 0),
                                "Status": status_text
                            }
                    except asyncio.TimeoutError:
                        # Continue waiting if no message arrived in this 1-second chunk
                        continue 
                    except json.JSONDecodeError:
                        continue # Ignore badly formatted messages
                        
        except Exception as e:
            # Actually print the error so you know if your API key or connection is failing!
            print(f"AISStream Connection Error: {e}") 
            
        return pd.DataFrame(ships.values()) if ships else pd.DataFrame()

    def get_distance(self, lat2, lon2):
        lat1, lon1 = self.hydro_pos
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return round(2 * asin(sqrt(a)) * 6371, 2)

# --- MODULE 4: HABITAT ANALYTICS ---
class AnalyticsEngine:
    @staticmethod
    def model_footprint(sl, threshold=100):
        if sl <= threshold: return 0.0
        d = np.linspace(0.001, 40.0, 2000)
        tl = 20 * np.log10(d * 1000)
        masking = np.where((sl - tl) <= threshold)[0]
        return d[masking[0]] if len(masking) > 0 else 40.0

    def process_recovery(self, sog):
        sog = float(sog)
        # SOG Zero Fix
        if sog <= 0.1: return 0.0, 0.0
            
        sl_now = 180 + (20 * np.log10(sog/15))
        target_speed = VesselSpeed.ECHO_SLOW.value
        sl_slow = sl_now - (60 * np.log10(sog/target_speed) if sog > target_speed else 0)
        
        r1, r2 = self.model_footprint(sl_now), self.model_footprint(sl_slow)
        return round(sl_now, 1), round(max(0.0, np.pi * (r1**2 - r2**2)), 2)

# --- USER INTERFACE ---
def run_app():
    st.set_page_config(page_title="OrcaSafe Tactical Interface", layout="wide")
    st.title("OrcaSafe: Acoustic Mitigation Interface")
    st.markdown("---")

    if 'init' not in st.session_state:
        st.session_state.fetcher = QuiltS3Fetcher()
        st.session_state.acoustic = AcousticEngine("orca_safe_brain.pth")
        st.session_state.maritime = MaritimeEngine("75bacd06abfea55882806410bd628919ae733cba")
        st.session_state.analytics = AnalyticsEngine()
        st.session_state.init = True

    st.sidebar.subheader("System Control Panel")
    mode = st.sidebar.radio("Input Source", ["Live Stream", "Historical Baseline"])
    ship_mode = st.sidebar.radio("AIS Data Source", ["Live AIS Stream", "Tactical Dataset (Ever Summit)"])
    refresh_rate = st.sidebar.slider("Refresh Interval (s)", 5, 30, 15)

    # Section 1: Acoustic Analysis
    img, timestamp = None, None
    if mode == "Live Stream":
        img, timestamp = st.session_state.fetcher.get_latest_spectrogram()
        st.caption(f"Telemetry Synchronized: {timestamp}" if timestamp else "Synchronizing Signal...")
    else:
        category = st.sidebar.selectbox("Signal Profile", ["Orca", "Mixed", "Noise"])
        path = f"Data/InferenceData/MixedInference/{category}"
        if os.path.exists(path) and os.listdir(path):
            img = Image.open(os.path.join(path, random.choice([f for f in os.listdir(path) if not f.startswith('.')]))).convert("RGB")
            st.caption(f"Baseline Source: {category}")

    if img:
        st.subheader("I. Acoustic Intelligence Analysis")
        col_img, col_metrics = st.columns([2, 1])
        col_img.image(img, use_container_width=True)
        
        prob = st.session_state.acoustic.run_inference(img)
        orca_detected = prob > 0.85
        
        with col_metrics:
            st.metric("Detection Confidence", f"{prob:.2%}")
            if orca_detected:
                st.error("ALERT: SRKW DETECTION CONFIRMED")
            else:
                st.info("STATUS: NO SRKW Detected")

        # --- CONDITIONAL TRIGGER: AIS & MITIGATION ---
        st.markdown("---")
        if orca_detected:
            st.subheader(f"II. Mitigation Analysis: Active Response Mode")
            
            if ship_mode == "Live AIS Stream":
                df = asyncio.run(st.session_state.maritime.fetch_live_ais())
            else:
                df = pd.DataFrame([{
                    "Name":"Ever Summit",
                    "SOG":17.2,
                    "latitude":48.12,
                    "longitude":-122.75,
                    "Status": "Underway (Engine)"
                }])
            
            if not df.empty:
                df['Distance (km)'] = df.apply(lambda x: st.session_state.maritime.get_distance(x['latitude'], x['longitude']), axis=1)
                phys = df.apply(lambda x: st.session_state.analytics.process_recovery(x['SOG']), axis=1)
                df[['Source Level (dB)', 'Reclaimed Area (km²)']] = pd.DataFrame(phys.tolist(), index=df.index)
                
                st.metric("Total Cumulative Habitat Recovery", f"{df['Reclaimed Area (km²)'].sum():.1f} km²")
                st.table(df[['Name', 'Status', 'SOG', 'Distance (km)', 'Source Level (dB)', 'Reclaimed Area (km²)']])
                
                # --- MAP HIGHLIGHTING LOGIC ---
                # Vessel Markers (Red)
                vessel_map_df = df[['latitude', 'longitude']].copy()
                vessel_map_df['color'] = '#FF0000' 
                
                # Hydrophone Marker (Blue)
                hydro_point = pd.DataFrame([{
                    'latitude': 48.1357, 
                    'longitude': -122.7597, 
                    'color': '#0000FF' 
                }])
                
                final_map_df = pd.concat([vessel_map_df, hydro_point], ignore_index=True)
                st.map(final_map_df, color='color', size=20)
                
                if st.button("Execute Mitigation Protocol"):
                    st.success("Protocol Signal Transmitted to Registered Vessels")
            else:
                st.info("No vessel-based masking threats detected in sector.")
        else:
            st.subheader("II. Mitigation Analysis: Standby Mode")
            st.caption("Maritime telemetry and AIS correlation are suspended until a positive detection is confirmed.")

    if mode == "Live Stream":
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    run_app()