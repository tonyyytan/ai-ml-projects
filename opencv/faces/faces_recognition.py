import cv2 as cv
import numpy as np
import os as os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

DIR = '/home/tony/ai-ml-projects/opencv/resources/Faces/train'

p = []
for i in os.listdir(DIR):
    p.append(i)
p.sort()  # Sort to ensure consistent label assignment (must match training order)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('/home/tony/ai-ml-projects/opencv/resources/Faces/val/ben_afflek/4.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Unidentified person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+w, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {p[label]} with confidence of {confidence}')

    cv.putText(img, str(p[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0,), thickness =2)

cv.imshow('Identified Person', img)

cv.waitKey(0)
