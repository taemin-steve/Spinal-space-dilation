import cv2 as cv 
import numpy as np

IMG_PATH ='./newData/7077.png'
imgInit = cv.imread(IMG_PATH,cv.IMREAD_GRAYSCALE)

cv.circle(imgInit, (48, 1180), 30, (256, 200, 0), 3)
cv.imshow('0',imgInit)
cv.imwrite("./processed/"+ str(j)+'.png', result)





cv.waitKey(0)