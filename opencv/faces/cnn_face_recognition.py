"""
PyTorch CNN Face Recognition Inference Script

Uses the trained ResNet18 model to recognize faces in new images.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2 as cv
import json
import os

# Configuration
MODEL_PATH = 'cnn_face_model.pth'
LABEL_MAP_PATH = 'label_map.json'
HAAR_CASCADE = 'haar_face.xml'
IMAGE_SIZE = 224
# Use CPU since CUDA compatibility is an issue
DEVICE = torch.device('cpu')
print("Using device: CPU")


def load_model(model_path, device):
    """
    Load the trained model and label mappings.
    """
    # Load label map
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    
    # Convert string keys to int
    label_map = {int(k): v for k, v in label_map.items()}
    num_classes = len(label_map)
    
    # Create model architecture with compatibility check
    try:
        # New API (torchvision >= 0.13)
        model = models.resnet18(weights=None)
    except TypeError:
        # Old API (torchvision < 0.13)
        model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load trained weights (always map to CPU for compatibility)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, label_map


def preprocess_face(image, detect_face=True):
    """
    Preprocess image for model input:
    1. Detect and crop face
    2. Resize to model input size
    3. Normalize using ImageNet statistics
    4. Convert to tensor
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Detect and crop face
    if detect_face:
        cascade = cv.CascadeClassifier(HAAR_CASCADE)
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = image_rgb[y:y+h, x:x+w]
        else:
            # No face detected, use center crop
            h, w = image_rgb.shape[:2]
            face_roi = image_rgb[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    else:
        face_roi = image_rgb
    
    # Convert to PIL Image
    pil_image = Image.fromarray(face_roi)
    
    # Apply same transforms as validation
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor_image = transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    return tensor_image


def predict_face(model, image_tensor, label_map, device):
    """
    Make prediction on a single face image.
    
    Returns:
        predicted_name: Name of predicted person
        confidence: Softmax probability (0-1, higher is better)
        all_probs: Dictionary with probabilities for all classes
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted = torch.max(probabilities, 1)
        predicted_idx = predicted.item()
        confidence_score = confidence.item()
        
        predicted_name = label_map[predicted_idx]
        
        # Get all probabilities
        all_probs = {}
        probs = probabilities[0].cpu().numpy()
        for idx, name in label_map.items():
            all_probs[name] = float(probs[idx])
    
    return predicted_name, confidence_score, all_probs


def recognize_from_image(image_path, model, label_map, device, threshold=0.5):
    """
    Recognize face from an image file.
    """
    # Load image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Preprocess
    image_tensor = preprocess_face(img)
    
    # Predict
    predicted_name, confidence, all_probs = predict_face(model, image_tensor, label_map, device)
    
    # Display results
    print(f"\nImage: {image_path}")
    print(f"Predicted: {predicted_name}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nAll probabilities:")
    for name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {prob:.2%}")
    
    # Visualize
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cascade = cv.CascadeClassifier(HAAR_CASCADE)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # Draw rectangle
        color = (0, 255, 0) if confidence >= threshold else (0, 0, 255)
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        # Draw label and confidence
        label_text = f"{predicted_name} ({confidence:.2%})"
        if confidence < threshold:
            label_text = f"Unknown ({confidence:.2%})"
        
        cv.putText(img, label_text, (x, y-10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv.imshow('Face Recognition', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return predicted_name, confidence


def main():
    """Main function"""
    # Load model
    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run cnn_face_train.py first to train the model.")
        return
    
    model, label_map = load_model(MODEL_PATH, DEVICE)
    print(f"Model loaded. Classes: {list(label_map.values())}")
    
    # Test on validation image
    test_image = '/home/tony/ai-ml-projects/opencv/resources/Faces/val/ben_afflek/2.jpg'
    
    if os.path.exists(test_image):
        recognize_from_image(test_image, model, label_map, DEVICE)
    else:
        print(f"Test image not found: {test_image}")
        print("Usage: python cnn_face_recognition.py")
        print("Or modify the test_image path in the code.")


if __name__ == '__main__':
    main()

