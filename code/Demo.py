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

# --- CONFIGURATION ---
FFMPEG_BINARY = "/opt/homebrew/bin/ffmpeg"
SHIP_TYPE_MAP = {
    70: "Cargo", 71: "Cargo (Haz A)", 80: "Tanker", 60: "Passenger", 0: "Unknown"
}

# --- MODULE 1: DATA FETCHER ---
class QuiltS3Fetcher:
    def __init__(self):
        self.bucket_name = "audio-orcasound-net"
        self.prefix = "rpi_port_townsend/hls/"
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def get_latest_spectrogram(self):
        try:
            lookback_start = int(time.time()) - (24 * 60 * 60)
            start_after_key = f"{self.prefix}{lookback_start}.ts"
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix, StartAfter=start_after_key)
            
            latest_obj = None
            for page in page_iterator:
                if "Contents" in page: latest_obj = page['Contents'][-1]
            
            if not latest_obj or latest_obj['Size'] < 1000: return None, None
            
            ts_file, wav_file = 'live_capture.ts', 'live_capture.wav'
            self.s3.download_file(self.bucket_name, latest_obj['Key'], ts_file)
            
            subprocess.run([FFMPEG_BINARY, '-v', 'error', '-i', ts_file, '-ar', '22050', '-ac', '1', wav_file, '-y'], capture_output=True)

            y, sr = librosa.load(wav_file, sr=22050)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=10000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(5, 5))
            plt.axis('off')
            librosa.display.specshow(S_dB, sr=sr, fmax=10000, ax=ax, cmap='magma')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).convert("RGB"), latest_obj['LastModified']
        except: return None, None

# --- MODULE 2: AI ENGINE ---
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

# --- MODULE 3: MARITIME ENGINE ---
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
                while time.time() - start_time < 5:
                    try:
                        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                        mmsi = data["MetaData"]["MMSI"]
                        if "PositionReport" in data["Message"]:
                            ships[mmsi] = {
                                "Name": data["MetaData"].get("ShipName", "Unknown").strip(),
                                "Type": SHIP_TYPE_MAP.get(data["MetaData"].get("ShipType", 0), "Other"),
                                "latitude": data["MetaData"]["latitude"],
                                "longitude": data["MetaData"]["longitude"],
                                "SOG": data["Message"]["PositionReport"].get("Sog", 0)
                            }
                    except: break
        except: pass
        return pd.DataFrame(ships.values()) if ships else pd.DataFrame()

    def get_distance(self, lat2, lon2):
        lat1, lon1 = self.hydro_pos
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return round(2 * asin(sqrt(a)) * 6371, 2)

# --- MODULE 4: PHYSICS ANALYTICS ---
class AnalyticsEngine:
    @staticmethod
    def model_footprint(sl, threshold=100):
        d = np.linspace(0.001, 40.0, 2000)
        tl = 20 * np.log10(d * 1000)
        masking = np.where((sl - tl) <= threshold)[0]
        return d[masking[0]] if len(masking) > 0 else 40.0

    def process_recovery(self, sog, dist):
        sog = float(sog)
        sl_now = 180 + (20 * np.log10(sog/15)) if sog > 0 else 0
        sl_slow = sl_now - (60 * np.log10(sog/7) if sog > 7 else 0)
        r1, r2 = self.model_footprint(sl_now), self.model_footprint(sl_slow)
        return round(sl_now, 1), round(max(0.1, np.pi * (r1**2 - r2**2)), 2)

# --- INTERFACE ---
def run_app():
    st.set_page_config(page_title="OrcaSafe Command Center", layout="wide")
    st.title("🐋 OrcaSafe: Continuous Strategic Command")

    if 'init' not in st.session_state:
        st.session_state.fetcher = QuiltS3Fetcher()
        st.session_state.acoustic = AcousticEngine("orca_safe_brain.pth")
        st.session_state.maritime = MaritimeEngine("75bacd06abfea55882806410bd628919ae733cba")
        st.session_state.analytics = AnalyticsEngine()
        st.session_state.init = True

    st.sidebar.header("🕹️ Controls")
    
    # Global Mode Selection
    mode = st.sidebar.radio("Command Mode", ["Live Stream", "Historical Analysis"])
    
    # UNLOCKED: Maritime Source is now available for BOTH modes
    st.sidebar.divider()
    st.sidebar.subheader("Maritime Settings")
    ship_mode = st.sidebar.radio("Maritime Data Source", ["Live AIS", "Historical Tactical (Ever Summit)"])
    
    if mode == "Historical Analysis":
        st.sidebar.divider()
        category = st.sidebar.selectbox("Signal Profile", ["Orca", "Mixed", "Noise"])
        refresh_rate = st.sidebar.slider("Cycle Rate (s)", 5, 30, 10)
    else:
        category = None
        refresh_rate = st.sidebar.slider("Cycle Rate (s)", 5, 30, 15)

    # 📡 PHASE 1: ACOUSTICS
    img, status = None, ""
    if mode == "Live Stream":
        img, timestamp = st.session_state.fetcher.get_latest_spectrogram()
        status = f"Live Signal: {timestamp}" if timestamp else "Syncing..."
    else:
        path = f"Data/InferenceData/MixedInference/{category}"
        if os.path.exists(path) and os.listdir(path):
            img = Image.open(os.path.join(path, random.choice([f for f in os.listdir(path) if not f.startswith('.')]))).convert("RGB")
            status = f"Historical Profile: {category}"

    if img:
        st.subheader("📡 Phase 1: Acoustic Intelligence")
        c1, c2 = st.columns([2, 1])
        c1.image(img, use_container_width=True, caption=status)
        prob = st.session_state.acoustic.run_inference(img)
        with c2:
            st.metric("AI Confidence", f"{prob:.1%}")
            if prob > 0.85: st.success("✅ BIOLOGICAL DETECTED")
            else: st.info("🔍 MONITORING...")

        # 🚢 PHASE 2: HABITAT RECOVERY
        st.divider()
        st.subheader(f"🚢 Habitat Recovery Analysis ({ship_mode})")
        
        # Determine which data to show
        if ship_mode == "Live AIS":
            df = asyncio.run(st.session_state.maritime.fetch_live_ais())
        else:
            # Tactical Demo Data (Ever Summit)
            df = pd.DataFrame([{"Name":"Ever Summit","SOG":17.2,"latitude":48.12,"longitude":-122.75}])
        
        if not df.empty:
            df['Distance (km)'] = df.apply(lambda x: st.session_state.maritime.get_distance(x['latitude'], x['longitude']), axis=1)
            phys = df.apply(lambda x: st.session_state.analytics.process_recovery(x['SOG'], x['Distance (km)']), axis=1)
            df[['Source Level', 'Reclaimed (km²)']] = pd.DataFrame(phys.tolist(), index=df.index)
            
            # RECOVERY DETAILS FIRST
            st.metric("Total Potential Habitat Reclaimed", f"{df['Reclaimed (km²)'].sum():.1f} km²")
            st.dataframe(df[['Name', 'SOG', 'Distance (km)', 'Source Level', 'Reclaimed (km²)']], use_container_width=True)
            
            # MAP SECOND
            st.map(df)
            
            if st.button("🚀 EXECUTE SLOWDOWN PROTOCOL"): st.balloons()
        else:
            st.info("Tactical zone clear of vessel masking threats.")

    # 🔄 AUTO-REFRESH
    if mode == "Live Stream":
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__": run_app()