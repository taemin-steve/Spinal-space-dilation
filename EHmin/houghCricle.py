# 허프 원 검출 (hough_circle.py)

import cv2
import numpy as np

img = cv2.imread('./c-arm 2023-05-09/7823.png')
# 그레이 스케일 변환 ---①
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 노이즈 제거를 위한 가우시안 블러 ---②
blur = cv2.GaussianBlur(gray, (3,3), 0)
# 허프 원 변환 적용( dp=1.2, minDist=30, cany_max=200 ) ---③
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 30, param1 = 100, param2 = 100, minRadius = 7, maxRadius = 100)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # 원 둘레에 초록색 원 그리기
        cv2.circle(img,(i[0], i[1]), i[2], (0, 255, 0), 2)
        # 원 중심점에 빨강색 원 그리기
        cv2.circle(img, (i[0], i[1]), 2, (255,255,255), 20)

# 결과 출력
img1 = cv2.resize(img, [700,700])
cv2.imshow('hough circle', img1)
cv2.imwrite('test2.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()