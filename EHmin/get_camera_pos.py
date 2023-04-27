import cv2
import numpy as np

'''
solvePnP : Nx3x1 or Nx1x3 형태의 object, image point를 가져야한다구 함.
그래서 (5,7,1,2) -> (35,1,2)
       (5,7,1,3) -> (35,1,3) 형태로 바꿔서 solvePNP 진행.
어차피 짝이 다 맞다구 생각해서 괜찮을거라구 생각함~!
'''


# ----------------------------------get 2D points by textfile---------------------------------------
imgp =[]

with open('./sub_info/circle_pos_noneBlank.txt', 'r') as file:
    count = 0
    for line in file:
        count += 1
        lineList = line.strip().split() 
        lineList=[float(i) for i in lineList] # str to float
        imgp.append([lineList])
imgp=np.array(imgp,dtype=np.float32)
#------------------------------------------------------------------------------------------------------

# ----------------------------------get 3D points by textfile--------------------------------------
objp = []

for i in range(5):
    with open('./sub_info/3D_coordinate/rigidbody_' + str(i +1) + '.txt', 'r') as file:
        flag = False
        count = 0 
        for line in file:
            if line.strip() == 'real_ref2':
                flag = True
                continue
            if(flag and count < 7):
                lineList = line.strip().split() 
                lineList=[float(i) for i in lineList] # str to float
                objp.append([lineList])
                count += 1 
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

print(objp[::])
# solvePnP 함수 호출
for i in range(5):
    retval, rvec, tvec = cv2.solvePnP(objp[7*i:7*i+7][::], imgp[7*i:7*i+7], K, dist,flags=cv2.SOLVEPNP_ITERATIVE)

    # rvec와 tvec 출력
    print("rvec = ", rvec)
    print("tvec = ", tvec)

