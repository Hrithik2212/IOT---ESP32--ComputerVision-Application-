import cv2 as cv
import mediapipe as mp 
import numpy as np 
import urllib
import os 
url = "http://192.168.134.102/capture"

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

filename="r1.avi"
fps=24.0
res='480p'

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv.VideoWriter_fourcc(*'XVID'),
}

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def get_dims(cap,res='480p'):  
    width,height=STD_DIMENSIONS[res]
    change_res(cap,width,height)
    return width,height

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

capture=cv.VideoCapture(0)
dims=get_dims(capture)
video_type=get_video_type(filename)
out=cv.VideoWriter(filename,video_type,fps,dims)



# capture  = cv.VideoCapture(0)

with mp_facedetector.FaceDetection(min_detection_confidence = 0.7) as fd:
  while True:
    imgResponse = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    frame = cv.imdecode(imgnp,-1)
    # isTrue , frame =  capture.read()
    frame2 = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = fd.process(frame2)
    cv.imshow("Video",frame)
    if results.detections:
        out.write(frame)
    cv.waitKey(100)
    if cv.waitKey(1) & 0xFF == ord("d"):
        break

capture.release()
out.release()
cv.destroyAllWindows()