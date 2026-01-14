import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ==============================
# 1. CONFIGURATION
# ==============================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

FOLDERS_TO_TEST = {
    "Orca_Vocal": "Data/InferenceData/MixedInference/Orca",
    "Mixed_Data": "Data/InferenceData/MixedInference/Mixed",
    "Noise": "Data/InferenceData/MixedInference/Noise"
}

MODEL_PATH = "orca_safe_brain.pth"
OUTPUT_PLOT = "snr_recall_proxy.png"

# ==============================
# 2. MODEL
# ==============================
model = models.mobilenet_v2()
model.classifier = nn.Sequential(
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ==============================
# 3. IMAGE TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 4. VISUAL SNR PROXY FUNCTION
# ==============================
def estimate_visual_snr(
    img_path,
    signal_band=(0.3, 0.6),   # Orca call band (fraction of freq axis)
    noise_band=(0.0, 0.2)     # Vessel noise band
):
    """
    Relative spectrogram SNR proxy using energy ratios.
    Returns SNR in dB (relative, not absolute).
    """
    img = Image.open(img_path).convert("L")
    arr = np.array(img.resize((224, 224)))

    F = arr.shape[0]

    sig_lo = int(signal_band[0] * F)
    sig_hi = int(signal_band[1] * F)
    noi_lo = int(noise_band[0] * F)
    noi_hi = int(noise_band[1] * F)

    signal_energy = np.mean(arr[sig_lo:sig_hi, :] ** 2)
    noise_energy = np.mean(arr[noi_lo:noi_hi, :] ** 2) + 1e-6

    return 10 * np.log10(signal_energy / noise_energy)

# ==============================
# 5. INFERENCE + DATA COLLECTION
# ==============================
results = []

for label_name, folder in FOLDERS_TO_TEST.items():
    if not os.path.exists(folder):
        continue

    # Ground truth definition
    true_label = 1 if label_name in ["Orca_Vocal", "Mixed_Data"] else 0

    for file in os.listdir(folder):
        if not file.lower().endswith(".png"):
            continue

        path = os.path.join(folder, file)

        # SNR proxy
        snr_proxy = estimate_visual_snr(path)

        # Model inference
        img_rgb = Image.open(path).convert("RGB")
        tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = model(tensor).item()
            pred_label = 1 if prob > 0.5 else 0

        results.append({
            "snr": snr_proxy,
            "true": true_label,
            "pred": pred_label,
            "correct": int(pred_label == true_label)
        })

df = pd.DataFrame(results)

# ==============================
# 6. RECALL VS SNR (ORCA ONLY)
# ==============================
df_orca = df[df["true"] == 1].copy()

# Bin SNR (2 dB bins)
df_orca["snr_bin"] = (df_orca["snr"] // 2) * 2

recall_by_bin = df_orca.groupby("snr_bin")["correct"].mean()

# ==============================
# 7. PLOT
# ==============================
plt.figure(figsize=(10, 6))

plt.plot(
    recall_by_bin.index,
    recall_by_bin.values,
    marker="o",
    linewidth=2
)

plt.fill_between(
    recall_by_bin.index,
    recall_by_bin.values,
    alpha=0.25
)

plt.axvline(
    x=0,
    linestyle="--",
    linewidth=1,
    label="High Masking Region"
)

plt.title("Orca Detection Recall vs Relative Spectrogram SNR", fontsize=14)
plt.xlabel("Relative Spectrogram SNR Proxy (dB)", fontsize=12)
plt.ylabel("Recall (Detection Probability)", fontsize=12)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
# Shade the Recovery Buffer Zone (0 to +2 dB)
plt.axvspan(
    0, 2,
    alpha=0.25,
    label="Recovery Buffer Zone"
)
plt.text(
    0.1, 0.75,
    "Unstable Detection\n(Physics-Limited)",
    fontsize=10,
    verticalalignment='center'
)
plt.show()

print(f"SNR–Recall plot saved to {OUTPUT_PLOT}")
