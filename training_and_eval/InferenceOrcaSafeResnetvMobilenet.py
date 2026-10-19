import os
import sys
from pathlib import Path
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import DETECTION_THRESHOLD
from core.models import load_mobilenet, load_resnet, standard_transform, device

TEST_DATA_PATH = PROJECT_ROOT / "Data" / "InferenceData" / "MixedInference" / "Mixed"

m_net_path = PROJECT_ROOT / "models" / "orca_safe_brain.pth"
r_net_path = PROJECT_ROOT / "models" / "orca_resnet_brain.pth"
m_net = load_mobilenet(str(m_net_path) if m_net_path.exists() else 'orca_safe_brain.pth').eval()
r_net = load_resnet(str(r_net_path) if r_net_path.exists() else 'orca_resnet_brain.pth').eval()

print(f"{'File Name':<25} | {'MobileNetV2':<18} | {'ResNet18':<18} | {'Match?'}")
print("-" * 75)

if TEST_DATA_PATH.exists():
    files = [f for f in TEST_DATA_PATH.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    for f in files[:100]:
        img = Image.open(f).convert('RGB')
        img_tensor = standard_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            m_prob = m_net(img_tensor).item()
            r_prob = r_net(img_tensor).item()

        # Labels evaluated at deployed threshold
        m_label = "ORCA" if m_prob > DETECTION_THRESHOLD else "NOISE"
        r_label = "ORCA" if r_prob > DETECTION_THRESHOLD else "NOISE"

        agreement = "" if m_label == r_label else " DISAGREE"
        m_conf = m_prob if m_prob > 0.5 else (1 - m_prob)
        r_conf = r_prob if r_prob > 0.5 else (1 - r_prob)

        print(f"{f.name[:25]:<25} | {m_label} ({m_conf:.1%}) | {r_label} ({r_conf:.1%}) | {agreement}")