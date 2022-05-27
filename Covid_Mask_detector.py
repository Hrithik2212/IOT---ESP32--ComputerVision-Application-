import cv2 as cv
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import urllib
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import urllib



model=load_model('C:\\Users\HRITHIK REDDY\Learning\Coincent Learn\AI\Mask detection\mask_detection_model.h5')
img_width,img_height=150,150
face_cascade=cv.CascadeClassifier("haar_face.xml")


img_count=0
font=cv.FONT_ITALIC
org=(1,1)
class_label=''
fontscale=1
color=(255,0,0)
thickness=2
url = "http://192.168.134.102/capture"


while True:
    imgResponse = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    frame = cv.imdecode(imgnp,-1)
    scale=50
    width = int(frame.shape[1]*scale/100)
    height =int(frame.shape[0]*scale/100)
    dim=(width,height)

    frame=cv.resize(frame,dim,interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,6)
    img_count2=0
    for x,y,w,h in faces:
        org=(x-10,y-10)
        img_count2+=1
        face_roi=frame[y:y+h,x:x+w]
        cv.imwrite("spam/input{}face{}.jpg".format(img_count,img_count2),face_roi)
        img=load_img("spam/input{}face{}.jpg".format(img_count,img_count2),target_size=(img_width,img_height))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        prediction=model.predict(img)
        if prediction==0:
            class_label="Mask"
            color=(0,255,0)
        else:
            class_label="No Mask"
            color=(0,0,255)

        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        cv.putText(frame,class_label,org,font,fontscale,color,thickness,cv.LINE_AA)
    
    cv.imshow("Face Mask Detection",frame)
    if cv.waitKey(1) & 0xff==ord("d"):
        break
cv.destroyAllWindows()
