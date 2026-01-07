import cv2 as cv

img = cv.imread('../resources/Photos/cats.jpg')
cv.imshow('cats', img)

#averaging to blur
average = cv.blur(img, (3, 3))
cv.imshow('average blur', average)

#gaussian blur
g_blur = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow('gaussian blur', g_blur)

#median blur
median = cv.medianBlur(img, 3)
cv.imshow('median blur', median)

#bilateral blur
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('bilateral blur', bilateral)


cv.waitKey(0)