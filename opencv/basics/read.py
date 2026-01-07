import cv2 as cv

# img = cv.imread('../resources/Photos/cat_large.jpg')

# if img is None:
#     print("Error: Could not load image from '../resources/Photos/cat.jpg'")
#     print("Make sure you're running from the 'basics' directory")
# else:
#     print(f"Image loaded successfully! Shape: {img.shape}")
#     cv.imshow('Cat', img)
#     print("Window should be displayed. Press any key to close...")
#     cv.waitKey(0)
#     cv.destroyAllWindows()

capture = cv.VideoCapture('../resources/Videos/dog.mp4')   

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)