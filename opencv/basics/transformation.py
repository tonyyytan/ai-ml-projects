import cv2 as cv
import numpy as np

img = cv.imread('../resources/Photos/park.jpg')
cv.imshow('park', img)

def translate(img, x, y):
    transMatrix = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMatrix, dimensions)

#-x -> left, -y -> up, x -> right, y -> down
translated = translate(img, -100, -100)
# cv.imshow('Translated', translated)

#rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    #rotating around the center
    if rotPoint is None:
        rotPoint = (width//2, height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)

#-45 -> counterclockwise, 45 -> clockwise
rotated = rotate(img, -45, (100,200))
cv.imshow('Rotated', rotated)

#resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

#flipping
#0 is flipping vertically
#1 is flipping horizontally
#-1 is flipping both
flipped = cv.flip(img, -1)
cv.imshow('Flipped', flipped)

#cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
