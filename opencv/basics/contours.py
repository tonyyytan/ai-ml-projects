import cv2 as cv
import numpy as np


img = cv.imread('../resources/Photos/cats.jpg')
cv.imshow('Cat', img)
blank = np.zeros(img.shape,dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('cat_gray', gray)

canny = cv.Canny(img, 125, 175)
cv.imshow('canny', canny)

# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow('blur', blur)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'canny {len(contours)} many contours found')

# canny_blur = cv.Canny(blur, 125, 175)
# cv.imshow('canny', canny_blur)

#binarizes an image -> turning it black or white only
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('thresh', thresh)

# contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(f'thresh {len(contours)} many contours found')

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('contours drawn', blank)



cv.waitKey(0)
