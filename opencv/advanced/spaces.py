import cv2 as cv

img = cv.imread('../resources/Photos/park.jpg')
cv.imshow('park', img)

#grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

#change to hsv
hsv  = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv', hsv)

#bgr to lab
lab = cv.cvtColor(img, cv.COLOR_LAB2LBGR)
cv.imshow('lab', lab)

#bgr to rgb
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#inversed color, opencv default as bgr
cv.imshow('rgb', rgb)

hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('hsv_bgr', hsv_bgr)

cv.waitKey(0)