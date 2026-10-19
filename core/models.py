import os
import torch
import torch.nn as nn
from torchvision import models, transforms

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Shared image transform
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_mobilenet(path=None, pretrained=False):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    elif path:
        print(f"WARNING: Model {path} not found.")
    return model.to(device)

def load_resnet(path=None, pretrained=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    elif path:
        print(f"WARNING: Model {path} not found.")
    return model.to(device)

class AcousticEngine:
    def __init__(self, model_path):
        self.device = device
        self.model = load_mobilenet(model_path).eval()
        self.transform = standard_transform

    def run_inference(self, img):
        if not img:
            return 0.0
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad(): 
            return self.model(tensor).item()
