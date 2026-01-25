import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random
import asyncio
import websockets
import json
from pyproj import Transformer

# --- AIS SHIP TYPE MAPPING ---
# Translates AIS numerical codes into human-readable categories
SHIP_TYPE_MAP = {
    70: "Cargo", 71: "Cargo (Haz A)", 72: "Cargo (Haz B)", 73: "Cargo (Haz C)", 
    74: "Cargo (Haz D)", 79: "Cargo (General)", 80: "Tanker", 81: "Tanker (Haz A)",
    82: "Tanker (Haz B)", 83: "Tanker (Haz C)", 84: "Tanker (Haz D)", 89: "Tanker (General)",
    60: "Passenger", 61: "Passenger (Haz A)", 62: "Passenger (Haz B)", 63: "Passenger (Haz C)",
    64: "Passenger (Haz D)", 69: "Passenger (General)"
}


# --- MODULE 1: ACOUSTIC ENGINE (AI DETECTION) ---
class AcousticEngine:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, path):
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(
            nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
        return model.to(self.device).eval()

    def run_inference(self, img):
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).item()

    @staticmethod
    def calculate_visual_snr(img, signal_band=(0.3, 0.6), noise_band=(0.0, 0.2)):
        """
        Relative spectrogram SNR proxy using energy ratios (Decibel Scale).
        Replaces linear mean subtraction with Logarithmic Ratio for better acoustic accuracy.
        """
        # Convert to grayscale and resize to match model input dimensions
        img_gray = np.array(img.convert('L').resize((224, 224)))
        
        # Determine frequency axis boundaries
        F = img_gray.shape[0]
        sig_lo, sig_hi = int(signal_band[0] * F), int(signal_band[1] * F)
        noi_lo, noi_hi = int(noise_band[0] * F), int(noise_band[1] * F)

        # Calculate Energy (Mean of squared pixel intensities)
        # Adding 1e-6 to avoid log(0) or division by zero errors
        signal_energy = np.mean(img_gray[sig_lo:sig_hi, :] ** 2)
        noise_energy = np.mean(img_gray[noi_lo:noi_hi, :] ** 2) + 1e-6

        # Return the Ratio in dB
        snr_db = 10 * np.log10(signal_energy / noise_energy)
        return round(float(snr_db), 1)

# --- MODULE 2: MARITIME ENGINE (GEOSPATIAL) ---
class MaritimeEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.bbox = [[47.5, -124.8], [49.0, -122.2]]
        self.hydro_pos = (48.1357, -122.7597) 
        self.transformer = Transformer.from_crs("epsg:4326", "epsg:32610", always_xy=True)
        self.h_x, self.h_y = self.transformer.transform(self.hydro_pos[1], self.hydro_pos[0])

    async def fetch_live_ais(self):
        ships = []
        try:
            async with websockets.connect("wss://stream.aisstream.io/v0/stream", open_timeout=12) as ws:
                msg = {"APIKey": self.api_key, "BoundingBoxes": [self.bbox], 
                       "FiltersShipTypes": [60, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]}
                await ws.send(json.dumps(msg))
                start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start < 4:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(raw)
                        if "MetaData" in data and "PositionReport" in data["Message"]:
                            # Extracting Type and mapping to string
                            type_code = data["MetaData"].get("ShipType", 0)
                            ships.append({
                                "Name": data["MetaData"]["ShipName"].strip(),
                                "Type": SHIP_TYPE_MAP.get(type_code, f"Other ({type_code})"),
                                "MMSI": data["MetaData"]["MMSI"],
                                "Lat": data["MetaData"]["latitude"],
                                "Lon": data["MetaData"]["longitude"],
                                "SOG": data["Message"]["PositionReport"]["Sog"]
                            })
                    except: break
        except: pass
        return pd.DataFrame(ships).drop_duplicates('MMSI') if ships else pd.DataFrame()

    def get_distance(self, lat, lon):
        s_x, s_y = self.transformer.transform(lon, lat)
        return round(float(np.sqrt((s_x - self.h_x)**2 + (s_y - self.h_y)**2) / 1000), 2)


class AnalyticsEngine:
    @staticmethod
    def model_acoustic_footprint(source_level_db, threshold_db=100):
        """
        Numerical Step-Walking Model: Finds the radius where ship noise
        meets the 100dB ambient floor using 20 Log R spreading.
        """
        # Search range up to 40km
        distances = np.linspace(0.001, 40.0, 2000) 
        
        # Spherical Spreading Model: TL = 20 * log10(Range_m)
        transmission_loss = 20 * np.log10(distances * 1000)
        received_levels = source_level_db - transmission_loss
        
        # Find index where noise drops below threshold
        masking_points = np.where(received_levels <= threshold_db)[0]
        
        if len(masking_points) > 0:
            return distances[masking_points[0]]
        return distances[-1] # Default to max range if still above threshold

    def apply_ross_law_and_geometry(self, sog, dist_to_hydro):
        """Calculates noise levels and habitat recovery area."""
        sog = float(sog)
        # 1. Source Level Calculation
        current_sl = 180 + (20 * np.log10(sog/15)) if sog > 0 else 0
        reduction = 60 * np.log10(sog/7) if sog > 7 else 0
        mitigated_sl = current_sl - reduction

        # 2. Radius Calculation
        r_transit = self.model_acoustic_footprint(current_sl)
        r_slowdown = self.model_acoustic_footprint(mitigated_sl)

        # 3. Area Calculation: Pi * (R1^2 - R2^2)
        area_transit = np.pi * (r_transit**2)
        area_slowdown = np.pi * (r_slowdown**2)
        reclaimed_km2 = area_transit - area_slowdown

        # 4. Limit results based on local distance
        if float(dist_to_hydro) > 50.0:
            return round(current_sl, 1), round(reduction, 1), 0.0

        return round(current_sl, 1), round(reduction, 1), round(max(0.1, reclaimed_km2), 2)

    @staticmethod
    def get_vessel_status(sog, dist):
        """
        Categorizes vessel threat level based on speed and proximity.
        Used for the tactical table status column.
        """
        sog, dist = float(sog), float(dist)
        if sog <= 0.1: 
            return "⚓ At Anchor/Docked"
        if sog > 7.0 and dist <= 15.0: 
            return "🔴 MASKING THREAT"
        return "🟢 Safe Transit"


# --- MODULE 4: INTERFACE ---
def run_app():
    st.set_page_config(page_title="OrcaSafe Command", layout="wide")
    st.title("🐋 OrcaSafe: Integrated Strategic Command Center")

    acoustic = AcousticEngine("orca_safe_brain.pth")
    maritime = MaritimeEngine("75bacd06abfea55882806410bd628919ae733cba")
    analytics = AnalyticsEngine()

    # UI Controls
    st.sidebar.header("🕹️ Controls")
    data_mode = st.sidebar.radio("Data Mode", ["Live AIS Stream", "Historical Demo"])
    category = st.sidebar.selectbox("Hydrophone Feed", ["Mixed", "Orca", "Noise"])
    if st.sidebar.button("🔄 Refresh Data Feed"): st.rerun()

    # PHASE 1: ACOUSTIC INTELLIGENCE
    st.subheader("📡 Phase 1: Acoustic Intelligence")
    path = f"Data/InferenceData/MixedInference/{category}"
    files = [f for f in os.listdir(path) if f.endswith(".png")]
    img_path = os.path.join(path, random.choice(files))
    img = Image.open(img_path).convert("RGB")
    
    col1, col2 = st.columns([2, 1])
    with col1: st.image(img, use_container_width=True)
    with col2:
        prob = acoustic.run_inference(img)
        snr = acoustic.calculate_visual_snr(img)
        st.metric("AI Confidence", f"{prob:.1%}")
        st.metric("Visual SNR Proxy", f"{snr} dB")
        is_detected, is_masked = prob > 0.85, snr < 2.0 and snr > -2.0
        if is_detected and is_masked: st.warning("⚠️ CRITICAL: MASKING ACTIVE")
        elif is_detected: st.success("✅ CLEAR SIGNAL")
        else: st.error("❌ NO DETECTION")

    # PHASE 2 & 3: TACTICAL RECOVERY
    if is_detected and is_masked:
        st.divider()
        st.subheader("🚢 Phase 2 & 3: Tactical Habitat Recovery")
        
        if data_mode == "Live AIS Stream":
            with st.spinner("Connecting to Live AIS Telemetry..."):
                df = asyncio.run(maritime.fetch_live_ais())
        else:
            df = pd.DataFrame([
                {"Name": "Ever Summit", "Type": "Cargo", "SOG": 17.2, "Lat": 48.12, "Lon": -122.75}
            ])

        if not df.empty:
            df['Dist'] = df.apply(lambda x: maritime.get_distance(x.Lat, x.Lon), axis=1)
            results = df.apply(lambda row: pd.Series(analytics.apply_ross_law_and_geometry(row['SOG'], row['Dist'])), axis=1)
            df[['SL', 'Savings', 'Reclaimed_km2']] = results
            df['Status'] = df.apply(lambda x: analytics.get_vessel_status(x.SOG, x.Dist), axis=1)

            st.metric("🌳 Total Habitat Reclaimed", f"{df['Reclaimed_km2'].sum():.1f} km²")

            # Final Table with Ship Type Column
            st.dataframe(
                df[['Name', 'Type', 'Status', 'SOG', 'Dist', 'SL', 'Reclaimed_km2']].style.apply(
                    lambda r: ['background-color: #ff4b4b; color: white' if r.Status == "🔴 MASKING THREAT" else '' for _ in r], axis=1
                ),
                use_container_width=True,
                column_config={
                    "Type": "Vessel Class",
                    "SOG": st.column_config.NumberColumn("Speed (kts)", format="%.1f"),
                    "Dist": st.column_config.NumberColumn("Dist (km)", format="%.1f"),
                    "SL": st.column_config.NumberColumn("Source Level (dB)", format="%.1f"),
                    "Reclaimed_km2": st.column_config.NumberColumn("Space Gained (km²)", format="%.1f")
                }
            )
            
            if st.button("🚀 EXECUTE SLOWDOWN"): st.balloons()
        else:
            st.info("No priority vessels found in the local basin.")

if __name__ == "__main__":
    run_app()