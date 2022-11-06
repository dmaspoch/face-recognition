# Face recognition using Python and OpenCV

This prototype uses the following libraries:
- Picamera2: Used for camera access to cameras connected using the flat ribbon cable directly to the Raspberry Pi board. (https://github.com/raspberrypi/picamera2)
- OpenCV: Used for face detection and recognition (https://github.com/opencv/opencv)
- eSpeakNG: Used for Text-To-Speech (https://github.com/espeak-ng/espeak-ng)

## Required libraries
Please install the required libraries using the following commands:

```
sudo apt install -y python3-picamera2 
sudo apt install -y python3-opencv
sudo apt install -y opencv-data

sudo apt-get install espeak-ng
pip install espeakng
```

You need to manually create 2 folders: "users" and "models" before running any of the Python files.

## Capturing a user face
Run the 01_create_users.py file and enter a name for the user. The user name will be used to create a folder under the "users" directory with the user images. The process captures 30 images by default, but this number can be changed using the MAX_CAPTURES constant.

## Training the face recognition model
Run the 02_face_training.py file. 
This process will output a "model.yml" file in the "models" folder.

## Performing face recognition
Update names array in line 20.

Run the 03_face_recognition.py file. 
The process will attempt to recognize the face using the camera, if the face was recorded in the model before.
