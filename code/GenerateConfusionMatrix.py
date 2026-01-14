import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. Setup Device & Path
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
script_dir = Path(__file__).resolve().parent
DATA_DIR = script_dir.parent / "Data" / "InferenceData" / "MixedInference" # Point to your folder

# 2. Recreate Model Architecture
def load_trained_model(model_path):
    model = models.mobilenet_v2() # Change to resnet18() if evaluating ResNet
    model.classifier = nn.Sequential(
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# 3. Data Loading with Binary Mapping
def get_inference_loader(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Apply the same binary logic: Noise -> 0, Mixed/Orca -> 1
    new_samples = []
    for path, target in dataset.samples:
        class_name = dataset.classes[target].lower()
        new_target = 0 if "noise" in class_name else 1
        new_samples.append((path, new_target))
    
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    
    return DataLoader(dataset, batch_size=32, shuffle=False)

# 4. Main Execution
if __name__ == "__main__":
    print(f"Loading data from {DATA_DIR}...")
    loader = get_inference_loader(DATA_DIR)
    
    print("Loading model...")
    model = load_trained_model('orca_safe_brain.pth')
    
    all_preds = []
    all_labels = []

    print("Running Inference...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 5. Calculate and Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Noise (0)', 'Orca Present (Masked Conditions Included)'], 
                yticklabels=['Noise (0)', 'Orca Present (Masked Conditions Included)'])
    
    plt.title('Final Validation Confusion Matrix: OrcaSafe System')
    plt.ylabel('Ground Truth (Actual)')
    plt.xlabel('AI Prediction')
    plt.savefig('confusion_matrix_final.png')
    print("Matrix saved as confusion_matrix_final.png")
    plt.show()