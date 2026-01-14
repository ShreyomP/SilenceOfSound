import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path

# This gets the folder where your script is currently located
script_dir = Path(__file__).resolve().parent
# This goes "up" one level and then into Code/Orca
TEST_DATA_PATH = script_dir.parent / "Data"  / "InferenceData" / "mixed"

IMAGE_SIZE = (224, 224)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. DEFINE & LOAD BOTH MODELS ---

def load_mobilenet(path):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_resnet(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

# Load both saved "brains"
m_net = load_mobilenet('orca_safe_brain.pth')
r_net = load_resnet('orca_resnet_brain.pth')

# --- 3. PREPROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. RUN COMPARISON LOOP ---
print(f"{'File Name':<25} | {'MobileNetV2':<18} | {'ResNet18':<18} | {'Match?'}")
print("-" * 75)

for filename in os.listdir(TEST_DATA_PATH)[:100]:
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(TEST_DATA_PATH, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            m_prob = m_net(img_tensor).item()
            r_prob = r_net(img_tensor).item()
        
        # Determine labels
        m_label = "ORCA" if m_prob > 0.5 else "NOISE"
        r_label = "ORCA" if r_prob > 0.5 else "NOISE"
        
        # Check if they agree
        agreement = "✅" if m_label == r_label else "❌ DISAGREE"
        
        # Display results with confidence
        m_conf = m_prob if m_prob > 0.5 else (1 - m_prob)
        r_conf = r_prob if r_prob > 0.5 else (1 - r_prob)
        
        print(f"{filename[:25]:<25} | {m_label} ({m_conf:.1%}) | {r_label} ({r_conf:.1%}) | {agreement}")