import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics import classification_report

# --- 1. CONFIGURATION ---
# Define your folders and their true labels
FOLDER_CONFIG = {
    "Data/InferenceData/MixedInference/Orca": 1,
    "Data/InferenceData/MixedInference/Noise": 0,
    "Data/InferenceData/MixedInference/Mixed": 1  # Assuming mixed contains orca signals
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. LOAD MODELS ---
def setup_mobilenet():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Linear(m.last_channel, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
    m.load_state_dict(torch.load('orca_safe_brain.pth', map_location=device))
    return m.to(device).eval()

def setup_resnet():
    r = models.resnet18(weights=None)
    r.fc = nn.Sequential(nn.Linear(r.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
    r.load_state_dict(torch.load('orca_resnet_brain.pth', map_location=device))
    return r.to(device).eval()

m_net = setup_mobilenet()
r_net = setup_resnet()

# --- 3. PROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

y_true = []
m_preds = []
r_preds = []

print("Running dual inference on labeled data...")

for folder, true_label in FOLDER_CONFIG.items():
    if not os.path.exists(folder): continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))]
    
    for f in files:
        img = Image.open(os.path.join(folder, f)).convert('RGB')
        img_t = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            m_prob = m_net(img_t).item()
            r_prob = r_net(img_t).item()
            
        y_true.append(true_label)
        m_preds.append(1 if m_prob > 0.5 else 0)
        r_preds.append(1 if r_prob > 0.5 else 0)

# --- 4. DISPLAY COMPARISON ---
print("\n" + "="*40)
print(" MOBILENET V2 PERFORMANCE")
print("="*40)
print(classification_report(y_true, m_preds, target_names=['Noise', 'Orca']))

print("\n" + "="*40)
print(" RESNET 18 PERFORMANCE")
print("="*40)
print(classification_report(y_true, r_preds, target_names=['Noise', 'Orca']))