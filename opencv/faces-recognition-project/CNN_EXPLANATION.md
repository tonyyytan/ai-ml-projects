# PyTorch CNN Face Recognition - Project Documentation

## 1. Project Overview
This project uses **Deep Learning (ResNet18)** to recognize faces from a large dataset. Unlike traditional methods (like LBPH) that struggle with lighting and angles, this model uses a Convolutional Neural Network (CNN) to learn robust facial features.

### **The "Brain" & The "Eyes"**
* **The Eyes (Haar Cascade):** A fast, lightweight algorithm that scans the image to find *where* the face is. It crops the face out.
* **The Brain (ResNet18):** A deep neural network that looks at the cropped face and decides *who* it is.

---

## 2. Project Structure
This project follows a decoupled architecture for modularity and scalability.

```text
faces-recognition-project/
├── model/                  # Stores the trained brain
│   └── cnn_face_model.pth  # The saved weights (the "knowledge")
├── resources/              # Shared tools used by both training & inference
│   ├── haar_face.xml       # The "Eyes" (Face Detector)
│   └── label_map.json      # The dictionary (0 = "Ben Affleck")
├── training/               # Logic to teach the model
│   ├── data_splits/        # CSV manifests (train.csv, val.csv, test.csv)
│   ├── cnn_face_train.py   # The teacher script
│   └── data_splitter.py    # The organizer script
├── inference/              # Logic to use the model
│   └── cnn_face_recognition.py
└── requirements.txt        # Dependency list