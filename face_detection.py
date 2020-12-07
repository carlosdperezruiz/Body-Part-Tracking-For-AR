import numpy as np
import cv2 as cv

faceCascade = cv.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    ret, img = cap.read()
    img = cv.flip(img, -1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv.imshow('video',img)
    k = cv.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv.destroyAllWindows()