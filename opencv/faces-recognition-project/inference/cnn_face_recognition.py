import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2 as cv
import json
import os
import pandas as pd
import random

# --- PATH CONFIGURATION ---
# Get the directory where this script is located (.../inference)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (.../faces-recognition-project)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# UPDATED: Load from 'model' folder
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'cnn_face_model.pth')

# Resources
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, 'resources', 'label_map.json')
HAAR_CASCADE_PATH = os.path.join(PROJECT_ROOT, 'resources', 'haar_face.xml')

# Test Data (Deep inside training folder)
TEST_CSV_PATH = os.path.join(PROJECT_ROOT, 'training', 'data_splits', 'test.csv')

IMAGE_SIZE = 224
DEVICE = torch.device('cpu') 

def load_model(model_path, device):
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}. Did you run training?")
        
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    
    try:
        model = models.resnet18(weights=None)
    except:
        model = models.resnet18(pretrained=False)
        
    model.fc = nn.Linear(model.fc.in_features, len(label_map))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Did you run training?")
        
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model, label_map

def preprocess_image(img_bgr):
    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Haar Cascade not found at {HAAR_CASCADE_PATH}")
    cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
    
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = img_rgb[y:y+h, x:x+w]
        face_rect = (x, y, w, h)
    else:
        h, w = img_rgb.shape[:2]
        face_roi = img_rgb[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        face_rect = None

    pil_img = Image.fromarray(face_roi)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0), face_rect

def test_random_batch(model, label_map, num_samples=10):
    if not os.path.exists(TEST_CSV_PATH):
        print(f"Error: Test data not found at {TEST_CSV_PATH}")
        return

    df = pd.read_csv(TEST_CSV_PATH)
    if len(df) < num_samples:
        samples = df
    else:
        samples = df.sample(n=num_samples)

    print(f"\n--- Testing {len(samples)} Random Images ---")
    print("Press any key on the image window to see the next one.\n")

    correct_count = 0
    
    for _, row in samples.iterrows():
        filepath = row['filepath']
        true_label = row['label_name']
        
        img = cv.imread(filepath)
        if img is None: continue
            
        img_tensor, face_rect = preprocess_image(img)
        img_tensor = img_tensor.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
        pred_label = label_map[predicted_idx.item()]
        conf_score = confidence.item()
        
        is_correct = (pred_label == true_label)
        if is_correct: correct_count += 1
        
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        
        if face_rect:
            x, y, w, h = face_rect
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
        text = f"Pred: {pred_label} ({conf_score:.1%})"
        cv.putText(img, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if not is_correct:
             cv.putText(img, f"Actual: {true_label}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"Actual: {true_label:<20} | Pred: {pred_label:<20} | Conf: {conf_score:.1%} | {'[OK]' if is_correct else '[X]'}")
        
        cv.imshow(f"Test: {true_label}", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    print(f"\nBatch Accuracy: {correct_count}/{len(samples)}")

if __name__ == '__main__':
    model, label_map = load_model(MODEL_PATH, DEVICE)
    test_random_batch(model, label_map, num_samples=10)