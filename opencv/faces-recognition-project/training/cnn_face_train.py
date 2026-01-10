"""
PyTorch CNN Face Recognition Training Script
Structure: Matches 'Final Version' folder layout
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2 as cv
import pandas as pd
import numpy as np

# --- PATH CONFIGURATION ---
# Get the directory where this script is located (.../training)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (.../faces-recognition-project)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Define Absolute Paths based on your Final Structure
TRAIN_CSV = os.path.join(CURRENT_DIR, 'data_splits', 'train.csv')
VAL_CSV = os.path.join(CURRENT_DIR, 'data_splits', 'val.csv')
HAAR_CASCADE_PATH = os.path.join(PROJECT_ROOT, 'resources', 'haar_face.xml')
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, 'resources', 'label_map.json')

# UPDATED: Save inside the 'model' folder
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'model', 'cnn_face_model.pth')

# Settings
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}")

class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv.imread(img_path)
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        # Load Cascade inside getitem to avoid multiprocessing crashes
        if not os.path.exists(HAAR_CASCADE_PATH):
             raise FileNotFoundError(f"Haar Cascade not found at {HAAR_CASCADE_PATH}")
        
        cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = img_rgb[y:y+h, x:x+w]
        else:
            h, w = img_rgb.shape[:2]
            face_roi = img_rgb[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        image = Image.fromarray(face_roi)
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(num_classes):
    try:
        model = models.resnet18(weights='IMAGENET1K_V1')
    except:
        model = models.resnet18(pretrained=True)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model():
    # 1. Load Data
    print("Loading data manifests...")
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: CSV files not found at {TRAIN_CSV}. Run data_splitter.py first.")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_paths = train_df['filepath'].tolist()
    train_labels = train_df['label_idx'].tolist()
    val_paths = val_df['filepath'].tolist()
    val_labels = val_df['label_idx'].tolist()

    person_names = sorted(train_df['label_name'].unique())
    num_classes = len(person_names)

    print(f"Classes: {num_classes}")

    # 2. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Loaders (num_workers=0 is crucial for Windows/OpenCV stability)
    train_dataset = FaceDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FaceDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model Setup
    model = create_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 5. Training Loop
    best_val_acc = 0.0
    print("\nStarting training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss_avg)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> New Best Model Saved ({val_acc:.1f}%)")

    # 6. Save Label Map
    label_map = {i: name for i, name in enumerate(person_names)}
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"\nDone! Best Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    train_model()