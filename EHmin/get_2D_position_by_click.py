import numpy as np
import cv2 as cv
import math

#-------------------- detector hyperparameter ------------------------------
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = False
params.filterByInertia = True
params.filterByColor = False
params.minArea = 3000 # The size of the blob filter to be applied. If the corresponding value is increased, small circles are not detected. 
params.maxArea = 80000
params.minCircularity = 0.01 # 1 >> it detects perfect circle. Minimum size of center angle
params.minInertiaRatio = 0.2 # 1 >> it detects perfect circle. short/long axis
params.minRepeatability = 3
params.minDistBetweenBlobs = 0.01
#-------------------------------------------------------------------------

#-------------------- text hyperparameter ------------------------------
fontFace = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 0, 0)
thickness = 2
lineType = cv.LINE_AA
#-------------------------------------------------------------------------
#------------------Mouse Click Event--------------------------------------
current_2D_pos = []
sorted_position = []

def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # print(current_2D_pos)
        distance_between_circle2mouse = []
        for circle_x, circle_y in current_2D_pos:
            distance_between_circle2mouse.append((circle_x - x)**2 + (circle_y - y)**2)
        min_value = min(distance_between_circle2mouse)
        min_index = distance_between_circle2mouse.index(min_value)
        print(current_2D_pos[min_index])
        sorted_position.append(current_2D_pos[min_index])
#---------------------------------------------------------------------------------------------

#------------------ create detector ------------------------------------------------
detector = cv.SimpleBlobDetector_create(params)
    
PATTERN_SIZE = (6,3) # 18 circle exist
UNIT_SIZE = 47.8125 # distance between circles // unit is millimeter

for j in range(21):
    # IMG_PATH ='./c-arm 2023-05-09/'+ str(j + 7802 )+ '.png'
    IMG_PATH ='./c-arm 2023-05-09/remake'+ str(7806)+ '.png'
    imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
    H, W = imgInit.shape[:2] 

    # detected circle
    keyPoints = detector.detect(imgInit)
    print(j)
    # visualize circle in original image
    
    
    for i in range(len(keyPoints)):
        keypoint = keyPoints[i]
        
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = keypoint.size
        r = int(math.floor(s / 2))
        
        # draw circle
        cv.circle(imgInit, (x, y), r, (256, 200, 0), 3) 
        cv.putText(imgInit, str(x) + "," + str(y),(x,y), fontFace, fontScale, color, thickness, lineType)
        current_2D_pos.append([x,y])
        
    # visualize 
    cv.imshow(str(j),imgInit) 
    cv.setMouseCallback(str(j), mouse_click)
    cv.waitKey(0)
    
    # Open a file for writing
    # with open("./EHmin/" + str(j) + "_2D_position.txt", "w") as f:
    #     for row in sorted_position:
    #         for item in row:
    #             f.write(str(item) + " ")
    #         f.write("\n")
            
    # save file by cv2.FileStorage()        
    # fs = cv.FileStorage("./EHmin/xml/" + str(j + 7817 )+ '.xml', cv.FILE_STORAGE_WRITE)
    fs = cv.FileStorage("./EHmin/xml/" + str(7806)+ '.xml', cv.FILE_STORAGE_WRITE)
    fs.write("my_data", str(np.array(sorted_position)))
    fs.release()

        
    current_2D_pos.clear()
    sorted_position.clear()
    
    
cv.waitKey(0)


