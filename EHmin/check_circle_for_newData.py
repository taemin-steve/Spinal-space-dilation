import numpy as np
import cv2 as cv
import math

params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = False
params.filterByInertia = True
params.filterByColor = False
params.minArea = 3000 #The size of the blob filter to be applied. If the corresponding value is increased, small circles are not detected. 
params.maxArea = 80000
params.minCircularity = 0.01 # 1 >> it detects perfect circle. Minimum size of center angle
params.minInertiaRatio = 0.2 # 1 >> it detects perfect circle. short/long axis
params.minRepeatability = 3
params.minDistBetweenBlobs = 0.01

######## for text
fontFace = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 0, 0)
thickness = 2
lineType = cv.LINE_AA
##########
# create detector
detector = cv.SimpleBlobDetector_create(params)
    
PATTERN_SIZE = (6,3) # 18 circle exist
UNIT_SIZE = 47.8125 # distance between circles // unit is millimeter

for j in range(4):
    IMG_PATH ='./newData/'+ str(j + 1 + 7080 )+ '.png'
    imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
    H, W = imgInit.shape[:2] 

    # detected circle
    keyPoints = detector.detect(imgInit)
    print(j)
    # visualize circle in original image
    for i in range(len(keyPoints)):
        keypoint = keyPoints[i]

        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = keypoint.size
        r = int(math.floor(s / 2))
        cv.circle(imgInit, (x, y), r, (256, 200, 0), 3) # draw circle
        cv.putText(imgInit, str(x) + "," + str(y),(x,y), fontFace, fontScale, color, thickness, lineType)
        print(x,y)
    imgInit = cv.resize(imgInit,[700,700])
    cv.imshow(str(j),imgInit) # visualize .
    cv.waitKey(0)
    
cv.waitKey(0)


