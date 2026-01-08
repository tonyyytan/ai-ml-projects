import cv2 as cv

img = cv.imread('../resources/Photos/cats.jpg')
cv.imshow('cats', img)

#simple thresholding
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('simple threshold', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('simple inverse threshold', thresh_inv)

#adaptive thresholding -> computer finds optimal threshold value
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('adaptive threshold', adaptive_thresh)



cv.waitKey(0)