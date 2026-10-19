import os
import sys
from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import DETECTION_THRESHOLD
from core.models import load_mobilenet, standard_transform, device

DATA_DIR = PROJECT_ROOT / "Data" / "Inference"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "Data" / "InferenceData" / "MixedInference"

def get_inference_loader(data_dir):
    dataset = datasets.ImageFolder(str(data_dir), transform=standard_transform)
    new_samples = []
    for path, target in dataset.samples:
        class_name = dataset.classes[target].lower()
        new_target = 0 if "noise" in class_name else 1
        new_samples.append((path, new_target))

    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    return DataLoader(dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print(f"Loading data from {DATA_DIR}")
    loader = get_inference_loader(DATA_DIR)

    model_path = PROJECT_ROOT / "models" / "orca_safe_brain.pth"
    model = load_mobilenet(str(model_path) if model_path.exists() else "orca_safe_brain.pth").eval()

    all_preds = []
    all_labels = []

    print(f"Running inference at deployed threshold ({DETECTION_THRESHOLD})")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > DETECTION_THRESHOLD).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Noise', 'Orca'],
                yticklabels=['Noise', 'Orca'])

    plt.title(f'Validation Confusion Matrix (Threshold {DETECTION_THRESHOLD})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    out_file = PROJECT_ROOT / 'results_and_graphs' / 'confusion_matrix_final.png'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Matrix saved to {out_file}")