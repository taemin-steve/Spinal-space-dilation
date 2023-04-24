import cv2
import numpy as np


# ----------------------------------get 2D points by textfile--------------------------------------
imgp =[]

with open('./sub_info/circle_pos_noneBlank.txt', 'r') as file:
    point2D = []
    count = 0
    for line in file:
        count += 1
        point2D.append([line.strip().split()])
        if count == 7:
            imgp.append(np.array(point2D).astype(np.float32))
            point2D = []
            count = 0
            
#------------------------------------------------------------------------------------------------------

# ----------------------------------get 3D points by textfile--------------------------------------
objp = []

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
        objp.append(np.array(point3D).astype(np.float32))
                
#------------------------------------------------------------------------------------------------------
# 카메라 파라미터
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
dist = np.array([0, 0, 0, 0], dtype=np.float32)

# solvePnP 함수 호출
retval, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)

# rvec와 tvec 출력
print("rvec = ", rvec)
print("tvec = ", tvec)