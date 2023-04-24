import cv2 as cv
import numpy as np
import math
# from circle import make_2D_points
import glob

# ----------------------------------get 2D points by textfile--------------------------------------
points2Ds =[]

with open('./sub_info/circle_pos_noneBlank.txt', 'r') as file:
    point2D = []
    count = 0
    for line in file:
        count += 1
        point2D.append([line.strip().split()])
        if count == 7:
            points2Ds.append(np.array(point2D).astype(np.float32))
            point2D = []
            count = 0
            
print(points2Ds) # to check shape of array 
#------------------------------------------------------------------------------------------------------

# ----------------------------------get 3D points by textfile--------------------------------------
points3Ds = []

for i in range(5):
    with open('./sub_info/3D_coordinate/rigidbody_' + str(i +1) + '.txt', 'r') as file:
        flag = False
        point3D = []
        count = 0 
        for line in file:
            if line.strip() == 'real_ref2':
                flag = True
                continue
            if(flag and count < 7):
                point3D.append(line.strip().split())
                print(point3D)
                count += 1 
        points3Ds.append(np.array(point3D).astype(np.float32))
                
print(points3Ds)
#------------------------------------------------------------------------------------------------------

    
# # ------------ Camera calibrate----------------------------------------------------------------------
# PATTERN_SIZE = (6,3) # 18 circle exist
UNIT_SIZE = 47.8125 # distance between circles // unit is millimeter

img_gray = cv.imread('./sub_info/2D_images/1.png', cv.IMREAD_GRAYSCALE)
H, W = img_gray.shape[:2] 

rms_err, mtx, dist, rvecs, tvecs = cv.calibrateCamera(points3Ds, points2Ds, (W, H), None, None)   
print("\nRMS:", rms_err)
print("camera intrinsic matrix:\n", mtx)      # 카메라 내부 매트릭스
print("distortion coefficients: ", dist.ravel()) # 왜곡 계수 출력


# --------------------- reprojection part ----------------------------------------------------
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
#-----------------------------------------------------------------------------------------



# '''
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (W,H), 1, (W,H))


# # undistort
# dst = cv.undistort(imgColor, mtx, dist, None,mtx)
# cv.imshow("dist",imgColor)
# cv.imshow("undist", dst)
# cv.waitKey(0)
# '''
