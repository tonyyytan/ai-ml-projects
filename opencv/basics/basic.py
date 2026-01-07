import cv2 as cv

img = cv.imread('../resources/Photos/park.jpg')
cv.imshow('park', img)

#grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#edge cascade -> canny
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

#dilated
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

#eroded
eroded = cv.erode(dilated, (7,7), iterations=3)
cv.imshow('Eroded', eroded)

cv.waitKey(0)