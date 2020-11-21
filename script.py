import numpy as np
import cv2 as cv

# cap = cv.VideoCapture(0)

# steps outlined in https://arxiv.org/pdf/1907.05281.pdf

def start():
    print("beginning program")
    featureExtractionUsingSURF()

    '''
    Supposed to bring up a video feed of camera, but not working on my computer

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv.imshow('frame',gray)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    '''

def backgroundSegmentation():
    print("Step 1: find silhouette of foreground")
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
    # https://www.pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/ -> has explanations

def featureExtractionUsingSURF(): # should take in an image (or N x N x 3 matrix that represents an image)
    print("Step 2: find features of image")
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html -> not sure this works with a venv
    # https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
    
    # examples: 
    img = cv.imread('test/test01.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite('sift_keypoints.jpg',img)
    
    # dont think this works on this version of opencv
    # surf = cv.xfeatures2d.SIFT_create(400)
    # kp, des = surf.detectAndCompute(img,None)
    # img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    # plt.imshow(img2),plt.show()

def filterKeypoints():
    print("Step 3: filter features within silhouette")

def manual():
    print("Step 4: ?")

def matchKeypointsFrameToFrame():
    print("Step 5: match keypoints")
    # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

if __name__ == "__main__":
    start()