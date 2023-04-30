import cv2 as cv
import numpy as np
import math
import CPDetect
import consts as const


IMG_PATH = "./newData/"
PATTERN_SIZE = (8,8) # 18 circle exist
UNIT_SIZE = 47.8125 # distance between circles // unit is millimeter

# 3D 좌표
pattern_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points *= UNIT_SIZE
points3Ds = [pattern_points for i in range(10)]
    
#2D 좌표
points2Ds=[]

# detector 생성.
cpdetector = CPDetect.CPdetect()

# Circle Grid 찾기.

imagesPath=[]
for i in range(7063,7080):
    path = IMG_PATH+str(i)+".png"
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    
    cpdetector.detect_circles(img)
    
    # keyPoints는 detector가 원을 감지한 것
    found, corners = cv.findCirclesGrid(img,PATTERN_SIZE,flags=cv.CALIB_CB_SYMMETRIC_GRID+cv.CALIB_CB_CLUSTERING,blobDetector=cpdetector.detecor)
    
    if found: # 찾았을 시,
        points2Ds.append(corners) # 2d 감지한거 넣기
        imagesPath.append(path)   #
        cv.drawChessboardCorners(img, PATTERN_SIZE, corners, found)
        
        cv.namedWindow("image",cv.WINDOW_NORMAL)
        cv.resizeWindow(winname="image",width=800,height=800)
        cv.imshow("image",img)
        cv.waitKey(0)
    
    else: # 못찾았을 시,
        print("can't find pattern, in "+ path ) 







'''
for idx,path in enumerate(imagesPath):
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    
    for j in range(len(points2Ds[idx])): # 원의 개수만큼 그린당.  
   # x = int(points2Ds[0][j][0][0])
   # y = int(points2Ds[0][j][0][1])
   # cv.circle(imgInit,(x,y),2,(0,0,255),3)
   # color=(0,0,255)
   # cv.putText(imgColor, str(j)+ "th point",(x,y), fontFace, fontScale, color, thickness, lineType)
    
    
        p_x = int(points2Ds[idx][j][0][0])
        p_y =int(points2Ds[idx][j][0][1])
        cv.circle(img ,(p_x,p_y),2,(0,255,0),3)
        color = (0, 255, 0)
        cv.putText(img , str(j)+ "th",(p_x,p_y), const.FONT_FACE, const.FONT_SCALE, color, const.THICKNESS, const.LINETYPE)
    cv.imwrite("./Mankyo/pattern_detect/patter_"+str(idx)+".png",img)

'''


# for i in range(7063,7080):
#     img = cv.imread(IMG_PATH+str(i)+".png",cv.IMREAD_GRAYSCALE)
#     # keyPoints는 detector가 원을 감지한 것
#     keyPoints = detector.detect(img)
    
#     # 원을 잘 감지했는지 visualiaze 하는 코드입니다.
#     for j in range(len(keyPoints)):
#         keypoint = keyPoints[j]

#         x = int(keypoint.pt[0])
#         y = int(keypoint.pt[1])
#         s = keypoint.size
#         r = int(math.floor(s / 2))
#         cv.circle(img, (x, y), r, (256, 200, 0), 3) # imgInit 파일에 원을 그려넣음.
#     cv.imwrite("./Mankyo/images/deteted_"+str(i)+".png",img)