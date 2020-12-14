import tkinter
from tkinter import Label, Button, StringVar
import cv2
from PIL import Image, ImageTk
import time
import numpy as np
import cv2 as cv

faceCascade = cv.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eyeCascade = cv.CascadeClassifier('xml/haarcascade_eye.xml')
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # create nose image
        load = Image.open("nose.png")
        noseImage = ImageTk.PhotoImage(load)
        self.nose = self.canvas.create_image(20, 20, image=noseImage)
        self.nose_pos = [20,20]
        self.noseBool = True

        # create eye image
        load = Image.open("eyeball.png")
        load = load.resize((128, 128))
        eyeImage = ImageTk.PhotoImage(load)
        self.eyeL = self.canvas.create_image(20, 20, image=eyeImage)
        self.eyeR = self.canvas.create_image(40, 40, image=eyeImage)
        self.eye_pos = [20,20,40,40]
        self.eyeBool = False

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
    
        # Button to turn on eyes and nose
        var = StringVar()
        var.set("Select an image")
        btnEye = Button(self.window, text="Eyes", width=50, command=self.EyeOverlap)
        btnNose = Button(self.window, text="Nose", width=50, command=self.NoseOverlap)
        btnEye.pack(anchor=tkinter.CENTER, expand=True, padx="10", pady="10")
        btnNose.pack(anchor=tkinter.CENTER, expand=True, padx="10", pady="10")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame)) 
            self.photo = self.getFaceFromFrame(frame)  
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.canvas.tag_raise(self.nose) if self.noseBool else self.canvas.tag_lower(self.nose)
            self.canvas.tag_raise(self.eyeL) if self.eyeBool else self.canvas.tag_lower(self.eyeL)
            self.canvas.tag_raise(self.eyeR) if self.eyeBool else self.canvas.tag_lower(self.eyeR)

        self.window.after(self.delay, self.update)
    
    def getFaceFromFrame(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if(self.noseBool):
            faces = faceCascade.detectMultiScale(
                gray,     
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(20, 20)
            )
            for (x,y,w,h) in faces:
                cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                self.canvas.move(self.nose, - self.nose_pos[0] + (x + w//2), - self.nose_pos[1] + (y + h//2))
                self.nose_pos = [x + w//2, y + h //2]
        if(self.eyeBool):
            eyes = eyeCascade.detectMultiScale(
                gray,     
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(20, 20)
            )
            left = True
            for (x,y,w,h) in eyes:
                cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                if(left):
                    self.canvas.move(self.eyeL, - self.eye_pos[0] + (x + w//2), - self.eye_pos[1] + (y + h//2))
                    self.eye_pos = [x + w//2, y + h //2] + self.eye_pos[2:]
                else:
                    self.canvas.move(self.eyeR, - self.eye_pos[2] + (x + w//2), - self.eye_pos[3] + (y + h//2))
                    self.eye_pos = self.eye_pos[:2] + [x + w//2, y + h //2]
                left = not left
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        img = Image.fromarray(img)              # convert the images to PIL format...
        img = ImageTk.PhotoImage(img)           # ...and then to ImageTk format
        return img
    def EyeOverlap(self):
        print("Hello Eye!")
        # cv.setMouseCallback('image',overlap_eye_function)
        self.eyeBool = not self.eyeBool
    def NoseOverlap(self):
        print("Hello Nose!")
        # cv.setMouseCallback('image',overlapNoseFunctions)
        self.noseBool = not self.noseBool

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(3,640) # set Width
        self.vid.set(4,480) # set Height
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
