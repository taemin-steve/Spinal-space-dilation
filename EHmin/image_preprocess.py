import cv2 as cv 
import numpy as np

# --- opening---
# for j in range(17):
#     IMG_PATH ='./newData/'+ str(7062 + j + 1)+ '.png'
#     imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
#     kernel = np.ones((5, 5), np.uint8)
#     result = cv.morphologyEx(imgInit, cv.MORPH_OPEN, kernel)
#     # cv.imshow("2", img_gray)
#     cv.imwrite("./processed/"+ str(j)+'.png', result)
# ---------------

# --- bulur---
# for j in range(17):
#     IMG_PATH ='./newData/'+ str(7062 + j + 1)+ '.png'
#     imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
#     result = cv.GaussianBlur(imgInit,(5,5),0)
#     # cv.imshow("2", img_gray)
#     cv.imwrite("./processed/"+ str(j)+'.png', result)
# ---------------
for j in range(17):
    IMG_PATH ='./newData/'+ str(7062 + j + 1)+ '.png'
    imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)
    result = cv.medianBlur(imgInit,21)
    # _, result = cv.threshold(result,200,255,cv.THRESH_BINARY)
    kernel = np.ones((9, 1), np.uint8)
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    cv.imwrite("./processed/"+ str(j)+'.png', result)



cv.waitKey(0)