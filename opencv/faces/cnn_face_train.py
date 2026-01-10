"""
PyTorch CNN Face Recognition Training Script

ML Algorithm: Transfer Learning with ResNet18 Convolutional Neural Network
Why it's the best choice:
1. Transfer Learning: Uses pretrained weights from ImageNet, allowing the model to leverage
   learned visual features (edges, textures, patterns) without training from scratch
2. ResNet18: Lightweight but powerful architecture with residual connections that prevent
   vanishing gradients and allow deeper learning
3. Fine-tuning: Adapts pretrained features specifically for face recognition task
4. Data Augmentation: Combats overfitting on small datasets by artificially expanding training data
5. End-to-end learning: Learns hierarchical features automatically (low-level edges → high-level face features)

How it works:
- Takes pretrained ResNet18 (trained on ImageNet)
- Replaces final classification layer with 5 outputs (one per person)
- Freezes early layers (keeps ImageNet features), fine-tunes later layers
- Uses data augmentation (rotation, flipping, color changes) to increase effective dataset size
- Trains using cross-entropy loss and Adam optimizer
- Saves model and label mappings for inference
"""

import os
import platform
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
TRAIN_DIR = '/home/tony/ai-ml-projects/opencv/resources/Faces/train'
VAL_DIR = '/home/tony/ai-ml-projects/opencv/resources/Faces/val'
HAAR_CASCADE = 'haar_face.xml'
IMAGE_SIZE = 224  # ResNet standard input size
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
# Use CPU since CUDA compatibility is an issue
DEVICE = torch.device('cpu')
print("Using device: CPU")
MODEL_SAVE_PATH = 'cnn_face_model.pth'
LABEL_MAP_PATH = 'label_map.json'



class FaceDataset(Dataset):
    """
    Custom dataset for face recognition.
    Loads images, detects faces, crops them, and applies transformations.
    """
    def __init__(self, image_paths, labels, transform=None, detect_faces=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.haar_cascade = cv.CascadeClassifier(HAAR_CASCADE) if detect_faces else None
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv.imread(img_path)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        # Detect and crop face
        if self.haar_cascade is not None:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            if len(faces) > 0:
                # Use first detected face
                x, y, w, h = faces[0]
                face_roi = img_rgb[y:y+h, x:x+w]
            else:
                # If no face detected, use center crop
                h, w = img_rgb.shape[:2]
                face_roi = img_rgb[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        else:
            face_roi = img_rgb
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(face_roi)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_paths_and_labels(base_dir):
    """
    Collect all image paths and corresponding labels from directory structure.
    Directory structure: base_dir/person_name/image.jpg
    """
    image_paths = []
    labels = []
    person_names = sorted(os.listdir(base_dir))
    
    for label, person_name in enumerate(person_names):
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_dir, img_name)
                image_paths.append(img_path)
                labels.append(label)
    
    return image_paths, labels, person_names


def create_model(num_classes):
    """
    Create ResNet18 model with transfer learning.
    
    Algorithm: ResNet18 (Residual Network with 18 layers)
    Architecture:
    - Conv layers: Extract hierarchical features (edges → textures → patterns → objects)
    - Residual blocks: Skip connections that allow gradients to flow through deep networks
    - Global Average Pooling: Reduces spatial dimensions
    - Fully Connected layer: Final classification layer (customized for our 5 classes)
    
    Transfer Learning Strategy:
    1. Load pretrained weights from ImageNet (1.2M images, 1000 classes)
    2. Freeze early layers (they learn general features)
    3. Fine-tune later layers (adapt to face-specific features)
    """
    # Load pretrained ResNet18 with compatibility for different torchvision versions
    try:
        # New API (torchvision >= 0.13)
        model = models.resnet18(weights='IMAGENET1K_V1')
    except TypeError:
        # Old API (torchvision < 0.13)
        try:
            model = models.resnet18(pretrained=True)
        except TypeError:
            # Fallback: no pretrained weights
            model = models.resnet18(weights=None)
            print("Warning: Could not load pretrained weights. Training from scratch.")
    
    # Freeze early layers (optional - can unfreeze for more fine-tuning)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Replace final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_model():
    """Main training function"""
    
    # Get training data
    print("Loading training data...")
    train_paths, train_labels, person_names = get_data_paths_and_labels(TRAIN_DIR)
    
    # Get validation data if available
    if os.path.exists(VAL_DIR):
        val_paths, val_labels, _ = get_data_paths_and_labels(VAL_DIR)
        print(f"Found {len(val_paths)} validation images")
    else:
        # Split training data for validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        print("Created validation split from training data")
    
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Classes: {person_names}")
    
    # Data augmentation for training (combats overfitting on small dataset)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(10),  # Rotate up to 10 degrees
        transforms.RandomHorizontalFlip(0.5),  # Randomly flip horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FaceDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FaceDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    # Use num_workers=0 on Windows/WSL to avoid multiprocessing issues
    num_workers = 0 if platform.system() == 'Windows' else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # Create model
    num_classes = len(person_names)
    model = create_model(num_classes)
    model = model.to(DEVICE)
    
    # Loss function: Cross-Entropy Loss (standard for classification)
    # Measures difference between predicted probabilities and true labels
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam (Adaptive Moment Estimation)
    # Combines benefits of momentum and adaptive learning rates
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (reduces LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("\nStarting training...")
    print("-" * 50)
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Get predictions
            loss = criterion(outputs, labels)  # Calculate loss
            
            # Backward pass (backpropagation)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate averages
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss_avg)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'person_names': person_names
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        print("-" * 50)
    
    # Save label mapping
    label_map = {i: name for i, name in enumerate(person_names)}
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")


if __name__ == '__main__':
    train_model()

