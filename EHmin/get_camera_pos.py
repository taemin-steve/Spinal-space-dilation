import cv2
import numpy as np


# ----------------------------------get 2D points by textfile--------------------------------------
imgp =[]

with open('./sub_info/circle_pos_noneBlank.txt', 'r') as file:
    point2D = []
    count = 0
    for line in file:
        count += 1
        lineList = line.strip().split() 
        lineList=[float(i) for i in lineList] # str to float
        point2D.append([lineList])
        if count == 7:
            imgp.append(np.array(point2D).astype(np.float32))
            point2D = []
            count = 0
imgp = np.array(imgp,dtype=np.float32)    
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
                lineList = line.strip().split() 
                lineList=[float(i) for i in lineList] # str to float
                point3D.append([lineList])
                #print(point3D)
                count += 1 
        objp.append(np.array(point3D).astype(np.float32))
objp = np.array(objp,dtype=np.float32) 

#------------------------------------------------------------------------------------------------------
# 카메라 파라미터
FX = round(616.2203979492188,3)
FY = round(616.5223388671875,3)
CX = round(327.8929443359375,3)
CY = round(236.85763549804688,3)

'''
K = [ fx, s, cx]
    [ 0, fy, cy]
    [ 0,  0, 1 ]
'''

K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
dist = np.array([0, 0, 0, 0], dtype=np.float32)

# solvePnP 함수 호출
retval, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist,flags=cv2.SOLVEPNP_ITERATIVE)

# rvec와 tvec 출력
print("rvec = ", rvec)
print("tvec = ", tvec)
