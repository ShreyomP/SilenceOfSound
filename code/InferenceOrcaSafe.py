import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. SETTINGS ---
TEST_FOLDER = 'inference/'  # Path to your folder of images
MODEL_PATH = 'orca_safe_brain.pth'
IMAGE_SIZE = (224, 224)

# --- 2. MODEL RECONSTRUCTION ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# Load weights and move to device (CPU/MPS/CUDA)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- 3. PREPROCESSING PIPELINE ---
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. LOOP THROUGH FOLDER ---
print(f"{'File Name':<30} | {'Prediction':<15} | {'Confidence'}")
print("-" * 60)

results = []

# List all files in the directory
for filename in os.listdir(TEST_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(TEST_FOLDER, filename)
        
        try:
            # Load and transform image
            img = Image.open(file_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(img_tensor)
                prob = output.item()
            
            # Determine Label
            label = "ORCA" if prob > 0.5 else "NOISE"
            confidence = prob if prob > 0.5 else (1 - prob)
            
            print(f"{filename:<30} | {label:<15} | {confidence:.2%}")
            results.append((filename, label, confidence))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# --- 5. SUMMARY ---
orca_count = sum(1 for _, label, _ in results if label == "ORCA")
print("-" * 60)
print(f"Total Processed: {len(results)}")
print(f"Orcas Detected: {orca_count}")
print(f"Ambient Noise:  {len(results) - orca_count}")