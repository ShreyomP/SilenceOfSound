import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import random
import time
import plotly.graph_objects as go

# --- 1. MODEL & PROCESSING LOGIC ---
DEVICE = torch.device("cpu")

@st.cache_resource
def load_orca_model(model_path):
    model = models.mobilenet_v2()
    model.classifier = nn.Sequential(
        nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 1), nn.Sigmoid()
    )
    # Ensure your .pth file is in the same folder
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def denoise_image(img_pil):
    """Subtracts the median 'noise floor' from the spectrogram."""
    img_array = np.array(img_pil).astype(np.float32)
    median = np.median(img_array, axis=1, keepdims=True)
    denoised = np.clip(img_array - median, 0, 255).astype(np.uint8)
    return Image.fromarray(denoised)

# --- 2. UI SETUP ---
st.set_page_config(page_title="OrcaSafe Command Center", layout="wide")
st.title("🐋 OrcaSafe: Integrated Mitigation Console")

# Sidebar Controls
st.sidebar.header("🕹️ System Controls")
use_denoising = st.sidebar.toggle("Enable Spectral Denoising", value=False)
data_source = st.sidebar.selectbox("Hydrophone Scenario:", ["Mixed (Masked/0dB)", "Orca (Clear)", "Background Noise"])
if st.sidebar.button("🔄 Capture New Sample"):
    st.session_state.slowdown_active = False # Reset on new sample
    st.rerun()

# --- 3. DUAL-VIEW SPECTROGRAMS ---
st.subheader("📡 Acoustic Signal Processing")
col_raw, col_proc = st.columns(2)

# Load Sample
FOLDER_MAP = {"Mixed (Masked/0dB)": "Mixed", "Orca (Clear)": "Orca", "Background Noise": "Noise"}
sample_dir = f"Data/InferenceData/MixedInference/{FOLDER_MAP[data_source]}"
files = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
img_file = random.choice(files)
raw_img = Image.open(os.path.join(sample_dir, img_file)).convert("RGB")

with col_raw:
    st.image(raw_img, caption="RAW INPUT (Vessel Noise Present)", use_container_width=True)

with col_proc:
    if use_denoising:
        processed_img = denoise_image(raw_img)
        st.image(processed_img, caption="DENOISED (Background Subtracted)", use_container_width=True)
        active_img = processed_img
    else:
        st.info("Denoising Inactive. Using raw signal for AI inference.")
        active_img = raw_img

# --- 4. AI INFERENCE ---
st.divider()
model = load_orca_model("orca_safe_brain.pth")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tensor = transform(active_img).unsqueeze(0)
with torch.no_grad():
    prob = model(tensor).item()

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("AI Detection Confidence", f"{prob:.2%}")
    is_detected = prob > 0.5
    if is_detected:
        st.success("✅ ORCA DETECTED")
    else:
        st.error("❌ NO DETECTION")

# --- 5. DYNAMIC MITIGATION ---
if is_detected and data_source == "Mixed (Masked/0dB)":
    with c2:
        st.warning("⚠️ CRITICAL: Masking Event Detected (0dB SNR). Action Required.")
        if 'slowdown_active' not in st.session_state:
            st.session_state.slowdown_active = False

        if st.button("🚀 INITIATE DYNAMIC SLOWDOWN", type="primary"):
            st.session_state.slowdown_active = True

    if st.session_state.slowdown_active:
        st.balloons()
        st.success("Target Speed: 7.0 Knots Achieved. SNR recovered by +12dB.")

    # Visualization
    st.subheader("🗺️ Habitat Recovery Map")
    m1, m2 = st.columns([2, 1])
    
    radius = 2.0 if st.session_state.slowdown_active else 8.0
    color = "rgba(0, 255, 0, 0.4)" if st.session_state.slowdown_active else "rgba(255, 0, 0, 0.4)"
    
    with m1:
        fig = go.Figure(go.Scattermapbox(
            lat=[48.5], lon=[-123.2],
            mode='markers',
            marker=go.scattermapbox.Marker(size=radius * 30, color=color),
            text=["Active Masking Zone"]
        ))
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=48.5, lon=-123.2), zoom=9.5),
            margin={"r":0,"t":0,"l":0,"b":0}, height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with m2:
        st.write("### Impact Summary")
        st.write(f"**Footprint Radius:** {radius} km")
        st.write(f"**SNR State:** {'Recovered' if st.session_state.slowdown_active else 'Masked'}")
        st.write(f"**Habitat Gain:** {'400%' if st.session_state.slowdown_active else '0%'}")