import cv2 as cv
import numpy as np

img = cv.imread('../resources/Photos/cats.jpg')
cv.imshow('cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

#laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('laplacian', lap)

#sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)

cv.imshow('x', sobelx)
cv.imshow('y', sobely)

combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('combined', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('canny', canny)

cv.waitKey(0)