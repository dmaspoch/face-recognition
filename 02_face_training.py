import os
import cv2
import numpy as np
from PIL import Image

USERS_DIR = "users/"

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

# Gets images and labels for training the face recognition model
def getTrainingData(userDirPath):
    imagePaths = [os.path.join(userDirPath, f) for f in os.listdir(userDirPath)]
    faceSamples = []
    for imagePath in imagePaths:
        # convert image to grayscale
        PIL_image = Image.open(imagePath).convert("L")
        # convert grayscale image to array
        numpy_image = np.array(PIL_image)
        faces = face_detector.detectMultiScale(numpy_image, 1.1, 5)
        for (x, y, w, h) in faces:
            faceSamples.append(numpy_image[y:y+h,x:x+w])
            
    return faceSamples

allFaces = []
allLabels = []
labelIndex = 1
with os.scandir(USERS_DIR) as it:
    for entry in it:
        if entry.is_dir():
            print(entry.name, labelIndex)
            userPath = USERS_DIR + entry.name
            # get face images for user from user folder
            userFaces = getTrainingData(userPath)
            # add images to list of all images
            allFaces.extend(userFaces)
            # each image must have a label
            for i in range(len(userFaces)):
                allLabels.append(labelIndex)
            labelIndex += 1

print("\n[INFO] Training recognizer with user images...")
recognizer.train(allFaces, np.array(allLabels))
# save model
recognizer.write("models/model.yml")

print("\n[INFO] {} faces trained".format(len(np.unique(allLabels))))