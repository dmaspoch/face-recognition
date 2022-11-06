import os
import cv2
from picamera2 import Picamera2

MAX_CAPTURES = 30
USERS_DIR = "users/"

# Get user id
user_id = input("Enter user name and press <return> ")

# Create user folder if it does not exist
userPath = USERS_DIR + user_id
pathExists = os.path.exists(userPath)
if not pathExists:
    os.makedirs(userPath)

face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Capture user images up to MAX_CAPTURES
count = 0
while True:
    im = picam2.capture_array()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))

    cv2.imshow("Camera", im)
    k = cv2.waitKey(100) % 256
    # Press ESC to exit
    if k == 27 or count >= MAX_CAPTURES:
        print("ESC pressed, exiting...")
        break
    elif k == 32:
        count += 1
        # Save captured image
        imgName = userPath + "/image_{}.jpg".format(count)
        cv2.imwrite(imgName, gray[y:y+h,x:x+w])
        print("{} saved".format(imgName))
              
cv2.destroyAllWindows()
