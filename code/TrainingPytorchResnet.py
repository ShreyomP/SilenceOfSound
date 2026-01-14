import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# 1. Hardware Selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available(): device = torch.device("cuda")
print(f"Device: {device}")

# 2. Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 
script_dir = Path(__file__).resolve().parent
DATA_DIR = script_dir.parent / "Data" / "Training" / "Orca"
print(f"Absolute path to data: {DATA_DIR}")

# 3. Data Augmentation
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Load Dataset & Remap Labels
# ImageFolder initially labels folders 0, 1, 2 alphabetically.
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

print(f"Original Folder Mapping: {full_dataset.class_to_idx}")

# REMAPPING LOGIC: Combine "Orca" and "Mixed" into Label 1, "Noise" into Label 0
new_samples = []
for path, target in full_dataset.samples:
    class_name = full_dataset.classes[target].lower()
    if "noise" in class_name:
        new_target = 0
    else:
        new_target = 1  # This handles both 'orca' and 'mixed' folders
    new_samples.append((path, new_target))

# Update the dataset with our new binary labels
full_dataset.samples = new_samples
full_dataset.targets = [s[1] for s in new_samples]

# Audit the new counts
counts = Counter(full_dataset.targets)
print(f"Final Class Counts: Noise (0): {counts[0]}, Orca/Mixed (1): {counts[1]}")

# Split into Train/Val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# 5. Model: ResNet18 Transfer Learning
model = models.resnet18(weights='DEFAULT')

# Freeze the "Backbone"
for param in model.parameters():
    param.requires_grad = False

# Replace ResNet's "fc" layer for binary classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model.to(device)

# 6. Optimizer & Loss
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

# 7. Training & Validation Loop
history = {'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    model.train()
    t_correct, t_total = 0, 0
    for inputs, labels in train_loader:
        # labels must be float and same shape as output [batch_size, 1]
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        t_correct += ((outputs > 0.5) == labels).sum().item()
        t_total += labels.size(0)

    # Validation
    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            v_correct += ((outputs > 0.5) == labels).sum().item()
            v_total += labels.size(0)

    val_acc = v_correct / v_total
    history['train_acc'].append(t_correct / t_total)
    history['val_acc'].append(val_acc)
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch+1}: Train Acc {history['train_acc'][-1]:.2f}, Val Acc {val_acc:.2f}")

# 8. Save
torch.save(model.state_dict(), 'orca_resnet_brain.pth')
print("ResNet18 Training Complete with Binary Remapping!")