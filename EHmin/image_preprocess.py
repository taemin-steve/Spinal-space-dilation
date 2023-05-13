import cv2 as cv 
import numpy as np
import math


#-------------------- detector hyperparameter ------------------------------
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = False
params.filterByInertia = True
params.filterByColor = False
params.minArea = 3000 # The size of the blob filter to be applied. If the corresponding value is increased, small circles are not detected. 
params.maxArea = 80000
params.minCircularity = 0.01 # 1 >> it detects perfect circle. Minimum size of center angle
params.minInertiaRatio = 0.2 # 1 >> it detects perfect circle. short/long axis
params.minRepeatability = 3
params.minDistBetweenBlobs = 0.01
#-------------------------------------------------------------------------

#-------------------- text hyperparameter ------------------------------
fontFace = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 0, 0)
thickness = 2
lineType = cv.LINE_AA
#-------------------------------------------------------------------------


for j in range(1):
    IMG_PATH ='./c-arm 2023-05-09/'+ str(7823)+ '.png'
    
    imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
    _, result = cv.threshold(imgInit, 0, 255, cv.THRESH_OTSU)
    # result = cv.adaptiveThreshold(imgInit ,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,2)
    cv.imshow('0', result)
    # _, result = cv.threshold(result,200,255,cv.THRESH_BINARY)
    
    kernel = np.ones((11,11), np.uint8)
    result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel)
    cv.imshow('1', result)
    kernel = np.ones((15,15), np.uint8)
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    cv.imshow('2', result)
    
    
    
    # cv.imwrite("./processed/"+ str(j)+'.png', result)
    
    detector = cv.SimpleBlobDetector_create(params)
    keyPoints = detector.detect(result)
    
    for i in range(len(keyPoints)):
        keypoint = keyPoints[i]
        
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = keypoint.size
        r = int(math.floor(s / 2))
        
        # draw circle
        cv.circle(imgInit, (x, y), r, (256, 200, 0), 3) 
        cv.putText(imgInit, str(x) + "," + str(y),(x,y), fontFace, fontScale, color, thickness, lineType)
    
    # cv.imshow('3', result)
    # cv.imshow('re', imgInit)
    
    


cv.waitKey(0)