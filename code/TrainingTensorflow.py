import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# 1. Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'Training/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Augmentation & Loading
# In PyTorch, we define a "Transform" pipeline
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Rescales 0-255 to 0.0-1.0 automatically
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard for MobileNet
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Split into 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Setup (Transfer Learning)
# Loading pre-trained MobileNetV2
model = models.mobilenet_v2(weights='DEFAULT')

# Freeze the "base_model" parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the "Head" (Top) of the model for Orca vs Noise
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model = model.to(device)
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 4. Training Loop (PyTorch requires writing this manually)
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

print("Starting Training: Teaching the AI the sound of the Sound...")

for epoch in range(15):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Simple Validation Loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_predicted = (outputs > 0.5).float()
            val_correct += (val_predicted == labels).sum().item()
            val_total += labels.size(0)

    # Store metrics for plotting
    history['accuracy'].append(correct/total)
    history['val_accuracy'].append(val_correct/val_total)
    history['loss'].append(running_loss/len(train_loader))
    history['val_loss'].append(val_loss/len(val_loader))
    
    print(f"Epoch {epoch+1}/15 - Loss: {history['loss'][-1]:.4f} - Acc: {history['accuracy'][-1]:.4f}")

# 5. Save the Model
torch.save(model.state_dict(), 'orca_safe_brain.pth')
print("\nSuccess! Model saved as 'orca_safe_brain.pth'")

# 6. Plotting (Same as your Keras code)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title('AI Learning Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('AI Learning Error (Loss)')
plt.legend()
plt.show()