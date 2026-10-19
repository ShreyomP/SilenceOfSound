import os
import sys
from pathlib import Path
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import DETECTION_THRESHOLD
from core.models import load_mobilenet, standard_transform, device

TEST_FOLDER = PROJECT_ROOT / "Data" / "InferenceData" / "MixedInference" / "Noise"
MODEL_PATH = PROJECT_ROOT / "models" / "orca_safe_brain.pth"

model = load_mobilenet(str(MODEL_PATH) if MODEL_PATH.exists() else "orca_safe_brain.pth").eval()

print(f"{'File Name':<30} | {'Prediction':<15} | {'Confidence'}")
print("-" * 60)

results = []

if TEST_FOLDER.exists():
    for f in TEST_FOLDER.iterdir():
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            try:
                img = Image.open(f).convert('RGB')
                img_tensor = standard_transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = model(img_tensor).item()

                label = "ORCA" if prob > DETECTION_THRESHOLD else "NOISE"
                confidence = prob if prob > 0.5 else (1 - prob)

                print(f"{f.name:<30} | {label:<15} | {confidence:.2%}")
                results.append((f.name, label, confidence))
            except Exception as e:
                print(f"Error processing {f.name}: {e}")

orca_count = sum(1 for _, label, _ in results if label == "ORCA")
print("-" * 60)
print(f"Total Processed: {len(results)}")
print(f"Orcas Detected: {orca_count}")
print(f"Ambient Noise:  {len(results) - orca_count}")