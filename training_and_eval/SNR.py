import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import librosa
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import DETECTION_THRESHOLD
from core.models import load_mobilenet, standard_transform, device

DATA_DIR = PROJECT_ROOT / "Data" / "InferenceData" / "MixedInference"
MODEL_PATH = PROJECT_ROOT / "models" / "orca_safe_brain.pth"
OUTPUT_PLOT = PROJECT_ROOT / "results_and_graphs" / "snr_recall_proxy.png"

FOLDERS_TO_TEST = {
    "Orca_Vocal": DATA_DIR / "Orca",
    "Mixed_Data": DATA_DIR / "Mixed",
    "Noise": DATA_DIR / "Noise"
}

model = load_mobilenet(str(MODEL_PATH) if MODEL_PATH.exists() else "orca_safe_brain.pth").eval()
transform = standard_transform

def estimate_spectrogram_snr(img_path):
    """
    Computes acoustic SNR ratio from the spectrogram representation.
    Casts to float64 to avoid integer overflow and uses correct frequency orientation.
    In the standardized spectrogram:
      - Rows 160-224 (bottom): low-frequency vessel noise band (200 Hz - 1.5 kHz)
      - Rows 50-140 (middle): orca fundamental whistle & call band (2 kHz - 8 kHz)
    """
    # Check for corresponding WAV file first for direct audio calculation
    wav_path = img_path.with_suffix('.wav')
    if wav_path.exists():
        try:
            y, sr = librosa.load(wav_path, sr=44100, duration=2.0)
            fft = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
            orca_band = (freqs >= 1500) & (freqs <= 8000)
            vessel_band = (freqs >= 50) & (freqs <= 600)
            sig_power = float(np.mean(fft[orca_band] ** 2)) if np.any(orca_band) else 1e-6
            noi_power = float(np.mean(fft[vessel_band] ** 2)) if np.any(vessel_band) else 1e-6
            return 10.0 * np.log10(max(1e-9, sig_power) / max(1e-9, noi_power))
        except Exception:
            pass

    # Spectrogram float calculation
    img = Image.open(img_path).convert("L")
    arr = np.array(img.resize((224, 224)), dtype=np.float64) / 255.0

    # Middle rows: orca call band (row 50 to 140)
    # Bottom rows: low frequency vessel noise (row 160 to 220)
    signal_energy = float(np.mean(arr[50:140, :] ** 2))
    noise_energy = float(np.mean(arr[160:220, :] ** 2)) + 1e-6

    return 10.0 * np.log10(signal_energy / noise_energy)

def evaluate_snr():
    results = []

    for label_name, folder in FOLDERS_TO_TEST.items():
        if not folder.exists():
            continue

        true_label = 1 if label_name in ["Orca_Vocal", "Mixed_Data"] else 0

        for file in folder.iterdir():
            if not file.suffix.lower().endswith(('.png', '.jpg')):
                continue

            snr_val = estimate_spectrogram_snr(file)

            img_rgb = Image.open(file).convert("RGB")
            tensor = transform(img_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = model(tensor).item()
                # Deployed production threshold (0.85)
                pred_label = 1 if prob > DETECTION_THRESHOLD else 0

            results.append({
                "snr": snr_val,
                "true": true_label,
                "pred": pred_label,
                "correct": int(pred_label == true_label)
            })

    if not results:
        print("No test data found for SNR evaluation.")
        return

    df = pd.DataFrame(results)
    df_orca = df[df["true"] == 1].copy()
    df_orca["snr_bin"] = (df_orca["snr"] // 2) * 2

    recall_by_bin = df_orca.groupby("snr_bin")["correct"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(recall_by_bin.index, recall_by_bin.values, marker="o", linewidth=2)
    plt.fill_between(recall_by_bin.index, recall_by_bin.values, alpha=0.25)
    plt.axvline(x=0, linestyle="--", linewidth=1, label="Masking Threshold (0 dB SNR)")
    plt.axvspan(0, 2, alpha=0.25, label="Transition Zone")

    plt.title(f"Orca Detection Recall vs SNR (Evaluated at {DETECTION_THRESHOLD} Threshold)", fontsize=13)
    plt.xlabel("Estimated Acoustic SNR (dB)", fontsize=12)
    plt.ylabel("Recall (Detection Rate)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

    print(f"SNR-Recall curve saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    evaluate_snr()
