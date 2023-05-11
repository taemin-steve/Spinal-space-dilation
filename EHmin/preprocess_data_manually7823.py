import numpy as np
import cv2 as cv
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


#------------------ create detector ------------------------------------------------
detector = cv.SimpleBlobDetector_create(params)
    
PATTERN_SIZE = (6,3) # 18 circle exist
UNIT_SIZE = 47.8125 # distance between circles // unit is millimeter

IMG_PATH ='./c-arm 2023-05-09/'+ str(7823)+ '.png'
imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
H, W = imgInit.shape[:2] 
# detected circle
keyPoints = detector.detect(imgInit)
# visualize circle in original image

for i in range(len(keyPoints)):
    keypoint = keyPoints[i]
    
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    s = keypoint.size
    r = int(math.floor(s / 2))
    
    print(x,y,r)
    # draw circle
    cv.circle(imgInit, (x, y), r, (256, 200, 0), 3) 
    
# visualize 
imgInit = cv.resize(imgInit, (700,700))
cv.imshow('4',imgInit) 

remake_image = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
cv.circle(remake_image, (47, 771), 42, (256, 200, 0), 3)
cv.circle(remake_image, (100, 1250), 44, (256, 200, 0), 3)
cv.imwrite("./c-arm 2023-05-09/remake7823.png", remake_image)

remake_image = cv.resize(remake_image, (700,700))
cv.imshow('reImage',remake_image) 

remake_image = cv.imread('./c-arm 2023-05-09/remake7823.png')
keyPoints = detector.detect(remake_image)
for i in range(len(keyPoints)):
    keypoint = keyPoints[i]
    
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    s = keypoint.size
    r = int(math.floor(s / 2))
    
    print(x,y,r)
    # draw circle
    cv.circle(remake_image, (x, y), r, (256, 200, 0), 3) 

remake_image = cv.resize(remake_image, (700,700))
cv.imshow('reDetect',remake_image) 





cv.waitKey(0)
        


