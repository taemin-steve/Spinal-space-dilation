import cv
import numpy as np

fs = cv2.FileStorage("./EHmin/data_blob.yml", cv2.FILE_STORAGE_READ)

# Read the variables from the file
g_minArea = int(fs.getNode("g_minArea").real())
g_maxArea = int(fs.getNode("g_maxArea").real())
g_minCircularity = int(fs.getNode("g_minCircularity").real())
g_minInertiaRatio = int(fs.getNode("g_minInertiaRatio").real())
g_minRepeatability = int(fs.getNode("g_minRepeatability").real())
g_minDistBetweenBlobs = int(fs.getNode("g_minDistBetweenBlobs").real())

l_minArea = int(fs.getNode("l_minArea").real())
l_maxArea = int(fs.getNode("l_maxArea").real())
l_minCircularity = int(fs.getNode("l_minCircularity").real())
l_minInertiaRatio = int(fs.getNode("l_minInertiaRatio").real())
l_minRepeatability = int(fs.getNode("l_minRepeatability").real())
l_minDistBetweenBlobs = int(fs.getNode("l_minDistBetweenBlobs").real())

# Close the file
fs.release()

# Set Blob Detector Params
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

