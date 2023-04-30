import cv2 as cv
import math

class CPdetect:
    def __init__(self):
        self.detecor=self.createBlobDetector() # detector 생성.
    
    def detect_circles(self,img):
        KeyPoints = self.detecor.detect(img)
        
        for i in range(len(KeyPoints)):
            keypoint = KeyPoints[i]
            s = keypoint.size
            radius = int(math.floor(s / 2)) #반지름 값.
    
            x = int(keypoint.pt[0]) #x 좌표
            y = int(keypoint.pt[1]) #y 좌표
        cv.circle(img, (x, y), radius, (256, 200, 0), 3) # draw circle
        #cv.putText(img, str(x) + "," + str(y),(x,y), fontFace, fontScale, color, thickness, lineType)
        
    def createBlobDetector(self):
        params = cv.SimpleBlobDetector_Params()
        params.filterByCircularity = True # 원에 가까운지로 판별.
        params.minCircularity = 0.8     # 얼마나 원에 가까운지! 1에 가까울 수록 원, 0에 가까울수록 삼각형 느낌.
        params.minInertiaRatio = 0.6    # 타원인지 원인지, 1에 가까울수록 원이다.
        
        # detector를 만듭니다.
        detector = cv.SimpleBlobDetector_create(params)
        
        return detector

   
'''
[params default Value]
.minRepeatabliity = 2  # blob 을 찾는데에 2번 반복.
.minDistBetweenBlobs = 10 # 두 블롭 사이의 최소 거리를 픽셀 단위로 지정.
.minCircularity = 0.8     # 얼마나 원에 가까운지! 1에 가까울 수록 원, 0에 가까울수록 삼각형 느낌.
.minInertiaRatio = 0.1    # 타원인지 원인지, 1에 가까울수록 원이다.

.filterByArea = True
.filterByCircularity = False
.filterByInertia = True
.filterByColor = True

'''