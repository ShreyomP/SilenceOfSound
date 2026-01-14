import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from collections import Counter # Added for the logbook audit

# 1. Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
script_dir = Path(__file__).resolve().parent
DATA_DIR = script_dir.parent / "Data" / "Training" / "Orca"
print(f"Absolute path to data: {DATA_DIR}")
 
# 1. Hardware Selection (Fixed for Mac/Apple Silicon)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# 2. Data Preparation & Binary Mapping
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
print(f"Original Folder Mapping: {full_dataset.class_to_idx}")

# --- BINARY REMAPPING LOGIC ---
# Unify Mixed and Orca into Class 1, Noise into Class 0
new_samples = []
for path, target in full_dataset.samples:
    class_name = full_dataset.classes[target].lower()
    if "noise" in class_name:
        new_target = 0
    else:
        new_target = 1  # Covers 'orca' and 'mixed' folders
    new_samples.append((path, new_target))

full_dataset.samples = new_samples
full_dataset.targets = [s[1] for s in new_samples]

# Audit for your logbook
counts = Counter(full_dataset.targets)
print(f"Final Class Counts: Noise (0): {counts[0]}, Orca/Mixed (1): {counts[1]}")
# ------------------------------

# Manually split (80% Train, 20% Val)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. The Model (Transfer Learning with MobileNetV2)
model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 4. Training Loop
epochs = 15
history = {'accuracy': [], 'val_accuracy': []}

print(f"Starting MobileNet AI Training on {DEVICE}...")

for epoch in range(epochs):
    model.train()
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        preds = (outputs > 0.5).float()
        running_corrects += torch.sum(preds == labels.data)
    
    train_acc = running_corrects.item() / len(train_dataset)
    history['accuracy'].append(train_acc)
    
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            val_corrects += torch.sum(preds == labels.data)
            
    val_acc = val_corrects.item() / len(val_dataset)
    history['val_accuracy'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}')

# 5. Save the Results
torch.save(model.state_dict(), 'orca_safe_brain.pth')
print("Model saved as orca_safe_brain.pth")

# 6. Plotting
plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Figure 4: MobileNetV2 Learning Curve (Binary Remapped)')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('learning_curve_mobilenet.png')
plt.show()