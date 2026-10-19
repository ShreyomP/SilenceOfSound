import os
import sys
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.models import load_mobilenet, standard_transform, device

model_path = PROJECT_ROOT / "models" / "orca_safe_brain.pth"
model = load_mobilenet(str(model_path) if model_path.exists() else 'orca_safe_brain.pth').eval()

def generate_saliency(img_path, save_path):
    preprocess = standard_transform

    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    output = model(input_tensor)
    score = output[0]
    score.backward()

    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = saliency.reshape(224, 224).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.resize((224, 224)))
    plt.title("Original Spectrogram")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='jet')
    plt.title("Saliency Map")
    plt.axis('off')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saliency map saved to {save_path}")

if __name__ == "__main__":
    img_cand = PROJECT_ROOT / "Data" / "Images" / "CleanOrca.png"
    out_cand = PROJECT_ROOT / "results_and_graphs" / "error_analysis_map.png"
    if img_cand.exists():
        generate_saliency(img_cand, out_cand)
    else:
        print(f"Sample image {img_cand} not found.")