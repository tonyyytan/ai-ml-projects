import cv2 as cv

img = cv.imread('../resources/Photos/cat.jpg')
cv.imshow('Cat', img)

def rescaleFrame(frame, scale=0.75):
    #works for images, videos and live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# capture = cv.VideoCapture('../resources/Videos/dog.mp4')   
# while True:
#     isTrue, frame = capture.read()
#     frame_resized = rescaleFrame(frame, scale=0.2)
#     cv.imshow('Original', frame)
#     cv.imshow('Video', frame_resized)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()

resized_image = rescaleFrame(img, scale = 0.2)
cv.imshow('Resized Image', resized_image)
cv.waitKey(0)

def changeRes(width, height):
    #works for live videos only
    capture.set(3, width)
    capture.set(4, height)

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    frame_resized = changeRes(frame, width=300, height=300)
    cv.imshow('Original', frame)
    cv.imshow('Video', frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()