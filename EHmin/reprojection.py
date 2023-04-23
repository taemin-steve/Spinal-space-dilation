import cv2 as cv
import numpy as np
import math
# from circle import make_2D_points
import glob

# ----------------------------------get 2D points--------------------------------------

# Define blob detector
# Parameter settings
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = False
params.filterByInertia = True
params.filterByColor = False
params.minArea = 35 # The size of the blob filter to be applied. If the corresponding value is increased, small circles are not detected. 
params.maxArea = 8000
params.minCircularity = 0.5 # 1 >> it detects perfect circle. Minimum size of center angle
params.minInertiaRatio = 0.5 # 1 >> it detects perfect circle. short/long axis
params.minRepeatability = 3
params.minDistBetweenBlobs = 0.01

# create detector
detector = cv.SimpleBlobDetector_create(params)
    
PATTERN_SIZE = (6,3) # 18 circle exist
UNIT_SIZE = 47.8125 # distance between circles // unit is millimeter

for j in range(5):
    IMG_PATH ='./sub_info/2D_images/'+ str(j + 1)+ '.png'
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
        cv.circle(imgInit, (x, y), r, (256, 200, 0), 3) # draw circle
    # cv.imshow(str(j),imgInit) # visualize 
# cv.waitKey(0)
############################################################################################################


#find 2D points gird################################################################################
# 카메라 paramerter는 한번만 구하면 됨
'''
points2Ds =[]

for i in range(5):
    IMG_PATH ='./sub_info/2D_images/'+ str(j + 1)+ '.png'
    img_gray = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE)
    # circle 좌표를 찾음
    ret, corners = cv.findCirclesGrid(img_gray,(6,3),flags=cv.CALIB_CB_SYMMETRIC_GRID+cv.CALIB_CB_CLUSTERING,blobDetector=detector)
    if ret:
        points2Ds.append(corners)
        points2Ds[i]=np.flip(points2Ds[i],0)


# -------------------- 원 잘 그려졌는지 판별하는 코드
fontFace = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2
lineType = cv.LINE_AA

imgColor = cv.imread(IMG_PATH)

for j in range(7): # 원의 개수만큼 그린당.
    # draw the center of the circle, it is original
    x = int(points2Ds[0][j][0][0])
    y = int(points2Ds[0][j][0][1])
    cv.circle(imgInit,(x,y),2,(0,0,255),3)
    color=(0,0,255)
    cv.putText(imgColor, str(j)+ "th point",(x,y), fontFace, fontScale, color, thickness, lineType)

cv.imshow("detect",imgColor)
cv.waitKey(0)
'''
#########################################################################################################

# 3D 좌표
pattern_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points *= UNIT_SIZE
points3Ds = [pattern_points for i in range(10)]

# # ------------ Camera calibrate
# rms_err, mtx, dist, rvecs, tvecs = cv.calibrateCamera(points3Ds, points2Ds, (W, H), None, None)   
# print("\nRMS:", rms_err)
# print("camera intrinsic matrix:\n", mtx)      # 카메라 내부 매트릭스
# print("distortion coefficients: ", dist.ravel()) # 왜곡 계수 출력



# fontFace = cv.FONT_HERSHEY_SIMPLEX
# fontScale = 1
# color = (0, 255, 0)
# thickness = 2
# lineType = cv.LINE_AA

# # reprojection output - 1번째 사진만.
# img_rePro = cv.imread('./circle_images/1_Color.png')
# mean_error=0
# # 3D 공간의 점을 2d 이미지로 reprojection.
# # imgpoints2는 tuple형식. shape : ( N, 1, 2 ), N은 3d point의 개수. 
# imgpoints2,_ = cv.projectPoints(points3Ds[0],rvecs[0],tvecs[0],mtx,dist)

# for j in range(len(imgpoints2)): # 원의 개수만큼 그린당.
#     # draw the center of the circle, it is original
#     '''
#     x = int(points2Ds[0][j][0][0])
#     y = int(points2Ds[0][j][0][1])
#     cv.circle(imgInit,(x,y),2,(0,0,255),3)
#     color=(0,0,255)
#     cv.putText(imgColor, str(j)+ "th point",(x,y), fontFace, fontScale, color, thickness, lineType)
#     '''
    
#     p_x = int(imgpoints2[j][0][0])
#     p_y =int(imgpoints2[j][0][1])
#     cv.circle(img_rePro ,(p_x,p_y),2,(0,255,0),3)
#     color = (0, 255, 0)
#     cv.putText(img_rePro , str(j)+ "th point",(p_x,p_y), fontFace, fontScale, color, thickness, lineType)

# cv.imshow('detected circles' + str(i + 1),img_rePro )
# cv.waitKey(0)
# error = cv.norm(points2Ds[0],imgpoints2,cv.NORM_L2)/len(imgpoints2)
# mean_error += error

# print("Total error : {0}".format(mean_error/len(points3Ds)))




# '''
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (W,H), 1, (W,H))


# # undistort
# dst = cv.undistort(imgColor, mtx, dist, None,mtx)
# cv.imshow("dist",imgColor)
# cv.imshow("undist", dst)
# cv.waitKey(0)
# '''
