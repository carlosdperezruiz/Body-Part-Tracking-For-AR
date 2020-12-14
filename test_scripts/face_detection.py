import numpy as np
import cv2 as cv
from tkinter import  Tk, Label, Button, StringVar
from PIL import Image, ImageTk

def EyeOverlap():
    print("Hello World!")
    # cv.setMouseCallback('image',overlap_eye_function)
def NoseOverlap():
    print("Hello Nose!")
    # cv.setMouseCallback('image',overlapNoseFunctions)
faceCascade = cv.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

global panelA
panelA = None

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
    # cv.imshow('video',img)
    root = Tk()

    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # convert the images to PIL format...
    img = Image.fromarray(img)
    # ...and then to ImageTk format
    img = ImageTk.PhotoImage(img)

    # if the panels are None, initialize them
    if panelA is None:
        # the first panel will store our original image
        panelA = Label(image=img)
        panelA.image = img
        panelA.pack(side="left", padx=10, pady=10)
    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=img)
        panelA.image = img

    # initialize the window toolkit along with the two image panels
    panelA = None
    panelB = None
    # create a button, then when pressed, will trigger a file chooser
    # dialog and allow the user to select an input image; then add the
    # button the GUI
    eye = Image.open("eyeball.png")
    phEye = ImageTk.PhotoImage(eye)
    nose = Image.open("nose.png")
    phNose = ImageTk.PhotoImage(nose)
    var = StringVar()
    label = Label( root, textvariable=var,padx="10", pady="10" )
    var.set("Select an image")
    label.pack()
    btnEye = Button(root, text="Eyes", image=phEye, command=EyeOverlap)
    btnNose = Button(root, text="Eyes", image=phNose, command=NoseOverlap)
    btnEye.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    btnNose.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    # kick off the GUI
    root.mainloop()
    
    k = cv.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv.destroyAllWindows()