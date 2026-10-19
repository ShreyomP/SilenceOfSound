import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.models import load_mobilenet, device

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = PROJECT_ROOT / "Data" / "Training"

# Transforms: Augmentation for training, pure normalization for validation
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SpectrogramDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def extract_recording_id(filepath):
    p = Path(filepath)
    name = p.stem
    if name.startswith("slice_"):
        parts = name.split("_")
        if len(parts) >= 3:
            return "_".join(parts[1:-1])
    return p.parent.name + "_" + name.split("_")[0]

def build_recording_split(raw_dataset, val_ratio=0.2):
    # Map samples to binary label: noise -> 0, orca/mixed -> 1
    # Group samples by original audio recording to prevent slice leakage
    rec_groups = defaultdict(list)
    for path, target in raw_dataset.samples:
        class_name = raw_dataset.classes[target].lower()
        new_target = 0 if "noise" in class_name else 1
        rec_id = extract_recording_id(path)
        rec_groups[rec_id].append((path, new_target))

    rng = np.random.RandomState(SEED)
    unique_recs = rng.permutation(list(rec_groups.keys()))
    val_count = max(1, int(len(unique_recs) * val_ratio))

    val_recs = set(unique_recs[:val_count])
    train_recs = set(unique_recs[val_count:])

    train_samples = [s for r in train_recs for s in rec_groups[r]]
    val_samples = [s for r in val_recs for s in rec_groups[r]]
    return train_samples, val_samples

def train():
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} not found.")
        return

    raw_dataset = datasets.ImageFolder(str(DATA_DIR))
    train_samples, val_samples = build_recording_split(raw_dataset, val_ratio=0.2)

    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    train_loader = DataLoader(SpectrogramDataset(train_samples, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpectrogramDataset(val_samples, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = load_mobilenet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    history = {'accuracy': [], 'val_accuracy': []}
    print(f"Training MobileNetV2 on {device}")

    for epoch in range(EPOCHS):
        model.train()
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data).item()

        train_acc = running_corrects / max(1, len(train_samples))
        history['accuracy'].append(train_acc)

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                val_corrects += torch.sum(preds == labels.data).item()

        val_acc = val_corrects / max(1, len(val_samples))
        history['val_accuracy'].append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    save_dir = PROJECT_ROOT / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "orca_safe_brain.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('MobileNetV2 Learning Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    out_plot = PROJECT_ROOT / "results_and_graphs" / "learning_curve_mobilenet.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot)
    plt.close()

if __name__ == "__main__":
    train()