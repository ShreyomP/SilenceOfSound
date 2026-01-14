import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Hardware Selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# 2. Recreate the EXACT architecture used in training
model = models.mobilenet_v2()

# This must match your TrainingPytorch.py exactly:
model.classifier = nn.Sequential(
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# 3. Load the weights
model.load_state_dict(torch.load('orca_safe_brain.pth', map_location=DEVICE))
model.to(DEVICE).eval()
print("Model loaded successfully!")

def generate_saliency(img_path, save_path):
    # Prepare Image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    # Forward Pass
    output = model(input_tensor)
    score = output[0]
    score.backward() # Calculate gradients

    # Get Saliency (max of absolute gradients across color channels)
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = saliency.reshape(224, 224).cpu().numpy()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.resize((224, 224)))
    plt.title("Original Spectrogram")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='jet')
    plt.title("Saliency Map (AI Focus Area)")
    plt.axis('off')

    plt.savefig(save_path)
    print(f"Saliency map saved to {save_path}")

# Run for a 'Mixed' image that confused the model
generate_saliency('Images/CleanOrca.png', 'error_analysis_map.png')