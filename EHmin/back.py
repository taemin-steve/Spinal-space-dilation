import cv2 as cv
import numpy as np
import math

# Get Params ----------------------------------------------------------------------------
fs = cv.FileStorage("./EHmin/data_blob_test.yml", cv.FILE_STORAGE_READ)

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
margin = int(fs.getNode("margin").real())

fs.release()

# Set Blob Detector -----------------------------------------------------------------------
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = False
params.filterByInertia = True
params.filterByColor = False

PATTERN_SIZE = (9,9) # 18 circle exist
UNIT_SIZE = 25 # distance between circles // unit is millimeter

# Image Setting-----------------------------------------------------------------------
IMG_PATH= "./newData/7065.png"
# IMG_PATH= "./newData/7081.png"
# IMG_PATH= "./EHmin/test.jpg"
IMG_PATH= "./EHmin/Phantom_only.png"
# IMG_PATH= "./EHmin/AP_view.png"


img = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE) # for houghCircle
H, W = img.shape[:2] 

# add mask 
img[0: 150,0: 125] = 0
img[0: 100,0: 250] = 0
img[0: 100,img.shape[0] - 100: img.shape[0]] = 0

img[img.shape[0] - 100: img.shape[0], 0: 260] = 0
img[img.shape[0] - 125: img.shape[0], 0: 200] = 0
img[img.shape[0] - 175: img.shape[0], 0: 120] = 0
cv.namedWindow('masked image', cv.WINDOW_NORMAL)
cv.imshow('masked image', img)


img_ori = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE) # save original image, draw final circle in here
img_ori = cv.cvtColor(img_ori, cv.COLOR_GRAY2BGR)

img_global_circle_detection = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE) # for draw global circle detection
img_global_circle_detection = cv.cvtColor(img_global_circle_detection, cv.COLOR_GRAY2BGR)

# Global Detection---------------------------------------------------------------------------------
params.minArea = g_minArea # The size of the blob filter to be applied. If the corresponding value is increased, small circles are not detected. 
params.maxArea = g_maxArea
params.minCircularity = g_minCircularity # 1 >> it detects perfect circle. Minimum size of center angle
params.minInertiaRatio = g_minInertiaRatio # 1 >> it detects perfect circle. short/long axis
params.minRepeatability = g_minRepeatability
params.minDistBetweenBlobs = g_minDistBetweenBlobs

detector = cv.SimpleBlobDetector_create(params) # set SimpleBlobDetector

keyPoints = detector.detect(img)

for i in range(len(keyPoints)):
    keypoint = keyPoints[i]
    x = round(keypoint.pt[0])
    y = round(keypoint.pt[1])
    s = keypoint.size
    r = round(s / 2)
    cv.circle(img_global_circle_detection, (x, y), r, (0, 0, 256), 2) # imgInit 파일에 원을 그려넣음.

cv.namedWindow('global_detection', cv.WINDOW_NORMAL)
cv.imshow('global_detection',img_global_circle_detection)
cv.waitKey(0)

# Local Detection --------------------------------------------------------------------------------------
# Blob Detection has enough accuracy
# params.minArea = l_minArea # The size of the blob filter to be applied. If the corresponding value is increased, small circles are not detected. 
# params.maxArea = l_maxArea
# params.minCircularity = l_minCircularity # 1 >> it detects perfect circle. Minimum size of center angle
# params.minInertiaRatio = l_minInertiaRatio # 1 >> it detects perfect circle. short/long axis
# params.minRepeatability = l_minRepeatability
# params.minDistBetweenBlobs = l_minDistBetweenBlobs

# detector = cv.SimpleBlobDetector_create(params) # set SimpleBlobDetector

# for i in range(len(keyPoints)):
#     keypoint = keyPoints[i]
#     x = round(keypoint.pt[0])
#     y = round(keypoint.pt[1])
#     s = keypoint.size
#     r = round(s / 2)
    
#     roi = img[y - (r+margin): y + (r+margin),x - (r+margin): x + (r+margin)]
    
#     k = detector.detect(roi)

# Camera calibrate -------------------------------------------------------------------------------------

# 2D Position 
points2Ds =[]

ret, corners = cv.findCirclesGrid(img,PATTERN_SIZE,flags=cv.CALIB_CB_SYMMETRIC_GRID+cv.CALIB_CB_CLUSTERING,blobDetector=detector)
if ret:
    points2Ds.append(corners)
    points2Ds = np.flip(points2Ds,0)
    
print(len(points2Ds))

# 3D Position
pattern_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points *= UNIT_SIZE
points3Ds = [pattern_points]

rms_err, mtx, dist, rvecs, tvecs = cv.calibrateCamera(points3Ds, points2Ds, (W, H), None, None)   

print("\nRMS:", rms_err)
print("camera intrinsic matrix:\n", mtx) # 카메라 내부 매트릭스
print("distortion coefficients: ", dist.ravel()) # 왜곡 계수 출력

# # Estimate R,T by solvePnp---------------------------------------------------
# retval, rvec, tvec = cv.solvePnP( points3Ds, points2Ds, mtx, dist, None )

# print(rvec, tvec)

