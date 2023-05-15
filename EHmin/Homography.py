import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10
# img1 = cv.imread('./c-arm 2023-05-09/7809.png', cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('./c-arm 2023-05-09/7817.png', cv.IMREAD_GRAYSCALE) # trainImage
img1 = cv.imread('./test5.png', cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('./test6.png', cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
# img1 = cv.resize(img1, [100,100])
# img2 = cv.resize(img2, [100,100])

sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

## mat을 바로 넣어주는 방시은 불가능 
# fs = cv.FileStorage("./EHmin/xml/" + str(7802) +'.txt', cv.FILE_STORAGE_FORMAT_YAML)
# node = fs.getNode('my_data')
# des1 = np.asarray(node.mat())
# print(des1.shape)
# fs.release()

# fs = cv.FileStorage("./EHmin/xml/" + str(7817) +'.txt', cv.FILE_STORAGE_FORMAT_YAML)
# node = fs.getNode('my_data')
# des2 = np.asarray(node.mat())
# fs.release()


matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []

for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    print(src_pts.shape)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
    

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()