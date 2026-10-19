import os
import sys
from pathlib import Path
import torch
from PIL import Image
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import DETECTION_THRESHOLD
from core.models import load_mobilenet, load_resnet, standard_transform, device

data_dir = PROJECT_ROOT / "Data" / "InferenceData" / "MixedInference"

FOLDER_CONFIG = {
    data_dir / "Orca": 1,
    data_dir / "Noise": 0,
    data_dir / "Mixed": 1
}

m_net_path = PROJECT_ROOT / "models" / "orca_safe_brain.pth"
m_net = load_mobilenet(str(m_net_path) if m_net_path.exists() else "orca_safe_brain.pth").eval()

r_net_path = PROJECT_ROOT / "models" / "orca_resnet_brain.pth"
r_net = load_resnet(str(r_net_path) if r_net_path.exists() else "orca_resnet_brain.pth").eval()

y_true = []
m_preds = []
r_preds = []

print(f"Running evaluation at deployed threshold: {DETECTION_THRESHOLD}")

for folder, true_label in FOLDER_CONFIG.items():
    if not folder.exists():
        continue
    files = [f for f in folder.iterdir() if f.suffix.lower() in ('.png', '.jpg')]

    for f in files:
        img = Image.open(f).convert('RGB')
        img_t = standard_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            m_prob = m_net(img_t).item()
            r_prob = r_net(img_t).item()

        y_true.append(true_label)
        m_preds.append(1 if m_prob > DETECTION_THRESHOLD else 0)
        r_preds.append(1 if r_prob > DETECTION_THRESHOLD else 0)

if y_true:
    print(f"\nMobileNetV2 (Threshold {DETECTION_THRESHOLD}):")
    print(classification_report(y_true, m_preds, target_names=['Noise', 'Orca']))

    print(f"\nResNet18 (Threshold {DETECTION_THRESHOLD}):")
    print(classification_report(y_true, r_preds, target_names=['Noise', 'Orca']))
else:
    print("No test data found in InferenceData folders.")