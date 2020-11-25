import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2()

# steps outlined in https://arxiv.org/pdf/1907.05281.pdf

def start():
    print("beginning program")
    # backgroundSegmentation()
    # featureExtraction('test/test01.jpg')
    # filterKeypoints()
    # manual()


    # Supposed to bring up a video feed of camera, but not working on my computer

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fgmask = backgroundSegmentation(frame)
        # Display the resulting frame
        # cv.imshow('frame',gray)
        img, kp, des = featureExtraction(fgmask)
        cv.imshow('frame1', img)
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

def backgroundSegmentation(frame):
    print("Step 1: find silhouette of foreground")
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
    # https://www.pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/ -> has explanations
    fgmask = fgbg.apply(frame)
    name = 'fgmask_frame_'+str(fgmask)+'.jpg'
    cv.imwrite(name,fgmask)
    return name

def featureExtraction(imgPath): # should take in an image (or N x N x 3 matrix that represents an image)
    print("Step 2: find features of image")
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html -> not sure this works with a venv
    # https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
    
    # examples: 
    img = cv.imread(imgPath)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite(imgPath[0:-4] + "_KEYPOINTS.jpg",img)
    return img, kp, des
    # print("a")
    # if img is None:
    #     print("Check file path")
    # cv.imshow('frame', img)
    # if cv.waitKey(10000):
    #     print("a.1")
    #     cv.destroyAllWindows()
    #     print("a.2")
    # print("b")
    
    # dont think this works on this version of opencv
    # surf = cv.xfeatures2d.SIFT_create(400)
    # kp, des = surf.detectAndCompute(img,None)
    # img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    # plt.imshow(img2),plt.show()

def filterKeypoints():
    print("Step 3: filter features within silhouette")
    frame1, frame1_kp, frame1_des = featureExtraction('test/test04.png')
    frame2, frame2_kp, frame2_des = featureExtraction('test/test05.png')
    frame3, frame3_kp, frame3_des = featureExtraction('test/test06.png')

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(frame1_des, frame2_des, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(frame1.shape[0], frame2.shape[0]), frame1.shape[1]+frame2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(frame1, frame1_kp, frame2, frame2_kp, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("MATCHES.jpg", img_matches)



def manual():
    print("Step 4: ?")
    img = cv.imread('test/test05.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv.imwrite("FACES.jpg", roi_color)
    

def matchKeypointsFrameToFrame():
    print("Step 5: match keypoints")
    # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

if __name__ == "__main__":
    start()