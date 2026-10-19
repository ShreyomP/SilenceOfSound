import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.signal import butter, sosfiltfilt

# Input folder with the wav files
SHARED_FOLDER_PATH = Path(__file__).resolve().parent.parent / "Wavfiles"
# Output folder, mirrors the input structure
BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Wavfiles" /"Images"

MAX_FILES_PER_FOLDER = 4

def save_spec(y, sr, fmin, fmax, output_path):
    # Saves normalized mel spectrogram
    try:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=224, fmin=fmin, fmax=fmax
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_norm, sr=sr, fmin=fmin, fmax=fmax, cmap='magma')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"      Error saving {output_path.name}: {e}")

def generate_all_versions(wav_path, target_dir):
    # Generates 5 filter variants
    y, sr = librosa.load(wav_path, sr=None, duration=2.0)
    stem = wav_path.stem
    
    # No filter
    save_spec(y, sr, 0, 20000, target_dir / f"NoFilter_{stem}.png")
    
    # 200Hz highpass
    sos200 = butter(4, 200, btype='highpass', fs=sr, output='sos')
    y200 = sosfiltfilt(sos200, y)
    save_spec(y200, sr, 200, 15000, target_dir / f"200hzpass_{stem}.png")
    
    # 200Hz with pre-emphasis
    y_opt = librosa.effects.preemphasis(y200, coef=0.97)
    save_spec(y_opt, sr, 200, 15000, target_dir / f"200hzopt_{stem}.png")
    
    # 500Hz highpass
    sos500 = butter(4, 500, btype='highpass', fs=sr, output='sos')
    y500 = sosfiltfilt(sos500, y)
    save_spec(y500, sr, 500, 15000, target_dir / f"500hzpass_{stem}.png")
    
    # 2000Hz highpass
    sos2000 = butter(4, 2000, btype='highpass', fs=sr, output='sos')
    y2000 = sosfiltfilt(sos2000, y)
    save_spec(y2000, sr, 2000, 15000, target_dir / f"2000hz_{stem}.png")

def run_local_orca_pipeline():
    print(f"Scanning {SHARED_FOLDER_PATH}\n")

    folder_counts = defaultdict(int)

    for root, dirs, files in os.walk(SHARED_FOLDER_PATH):
        path_obj = Path(root)

        # Relative path keeps the output structure
        rel_path = path_obj.relative_to(SHARED_FOLDER_PATH)

        wav_files = [f for f in files if f.lower().endswith('.wav')]

        for filename in wav_files:
            if folder_counts[str(rel_path)] >= MAX_FILES_PER_FOLDER:
                break

            input_wav = path_obj / filename
            target_dir = BASE_OUTPUT_DIR / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)

            print(f"Found {rel_path}: {filename}")
            print("  Generating 5 spectrogram versions")

            generate_all_versions(input_wav, target_dir)
            folder_counts[str(rel_path)] += 1

    print("\nDone")

if __name__ == "__main__":
    run_local_orca_pipeline()