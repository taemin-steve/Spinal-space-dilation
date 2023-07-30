import cv2 as cv
import numpy as np

# Get Params ----------------------------------------------------------------------------
fs = cv.FileStorage("./EHmin/data_blob.yml", cv.FILE_STORAGE_READ)

# Read the variables from the file
g_minArea = int(fs.getNode("g_minArea").real())
g_maxArea = int(fs.getNode("g_maxArea").real())
g_minCircularity = float(fs.getNode("g_minCircularity").real())
g_minInertiaRatio = float(fs.getNode("g_minInertiaRatio").real())
g_minRepeatability = int(fs.getNode("g_minRepeatability").real())
g_minDistBetweenBlobs = float(fs.getNode("g_minDistBetweenBlobs").real())

l_minArea = int(fs.getNode("l_minArea").real())
l_maxArea = int(fs.getNode("l_maxArea").real())
l_minCircularity = float(fs.getNode("l_minCircularity").real())
l_minInertiaRatio = float(fs.getNode("l_minInertiaRatio").real())
l_minRepeatability = int(fs.getNode("l_minRepeatability").real())
l_minDistBetweenBlobs = float(fs.getNode("l_minDistBetweenBlobs").real())

fs.release()

# Set Blob Detector -----------------------------------------------------------------------
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = False
params.filterByInertia = True
params.filterByColor = False

params.minArea = g_minArea # The size of the blob filter to be applied. If the corresponding value is increased, small circles are not detected. 
params.maxArea = g_maxArea
params.minCircularity = g_minCircularity # 1 >> it detects perfect circle. Minimum size of center angle
params.minInertiaRatio = g_minInertiaRatio # 1 >> it detects perfect circle. short/long axis
params.minRepeatability = g_minRepeatability
params.minDistBetweenBlobs = g_minDistBetweenBlobs

PATTERN_SIZE = (8,8) # 18 circle exist
UNIT_SIZE = 15.9375 # distance between circles // unit is millimeter

detector = cv.SimpleBlobDetector_create(params)

# Image Path---------------------------------------------------------------------------------

IMG_PATH= "./newData/7065.png"
img = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE) # for houghCircle
H, W = img.shape[:2] 

# add mask 
img[0: 150,0: 250] = 0
img[img.shape[0] - 100: img.shape[0], 0: 260] = 0
img[img.shape[0] - 175: img.shape[0], 0: 135] = 0
cv.imshow('masked image', img)


img_ori = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE) # save original image, draw final circle in here
img_ori = cv.cvtColor(img_ori, cv.COLOR_GRAY2BGR)

img_global_circle_detection = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE) # for draw global circle detection
img_global_circle_detection = cv.cvtColor(img_global_circle_detection, cv.COLOR_GRAY2BGR)

# keyPoints는 detector가 원을 감지
keyPoints = detector.detect(img)

# 원을 잘 감지했는지 visualiaze 하는 코드입니다.
for i in range(len(keyPoints)):
    keypoint = keyPoints[i]
    x = round(keypoint.pt[0])
    y = round(keypoint.pt[1])
    s = keypoint.size
    r = round(s / 2)
    cv.circle(img_global_circle_detection, (x, y), r, (0, 0, 256), 2) # imgInit 파일에 원을 그려넣음.

cv.imshow('global_detection',img_global_circle_detection)

cv.waitKey(0)