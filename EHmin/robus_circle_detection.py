import cv2
import numpy as np

# GET parameter --------------------------------------------------------
fs = cv2.FileStorage("./EHmin/data.yml", cv2.FILE_STORAGE_READ)


# Read the variables from the file
g_dp = int(fs.getNode("g_dp").real())
g_minDist = int(fs.getNode("g_minDist").real())
g_param1 = int(fs.getNode("g_param1").real())
g_param2 = int(fs.getNode("g_param2").real())
g_minRadius = int(fs.getNode("g_minRadius").real())
g_maxRadius = int(fs.getNode("g_maxRadius").real())
l_dp = int(fs.getNode("l_dp").real())
l_minDist = int(fs.getNode("l_minDist").real())
l_param1 = int(fs.getNode("l_param1").real())
l_param2 = int(fs.getNode("l_param2").real())
margin = int(fs.getNode("margin").real())
# Close the file
fs.release()


# Read image--------------------------------------------------------------------------------
img_path= "./newData/7065.png"
img_path= "./EHmin/test.jpg"
img_path= "./EHmin/Phantom_only.png"


img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # for houghCircle

# add mask 
img[0: 150,0: 250] = 0
img[img.shape[0] - 100: img.shape[0], 0: 260] = 0
img[img.shape[0] - 175: img.shape[0], 0: 135] = 0
cv2.namedWindow('masked image', cv2.WINDOW_NORMAL)
cv2.imshow('masked image', img)
cv2.waitKey()

img_ori = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # save original image, draw final circle in here
img_ori = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2BGR)

img_global_circle_detection = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # for draw global circle detection
img_global_circle_detection = cv2.cvtColor(img_global_circle_detection, cv2.COLOR_GRAY2BGR)

# check OTUS ---------------------------------------------------------------------------------------------
# _, img_OTSU = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# kernel = np.ones((5, 5), np.uint8) 
# img_OTSU = cv2.morphologyEx(img_OTSU, cv2.MORPH_OPEN, kernel)
# img_OTSU = 255 - img_OTSU
# # cv2.namedWindow('n', cv2.WINDOW_NORMAL)
# cv2.imshow('OTSI', img_OTSU)


# Global Circle Detection ---------------------------------------------------------------------------------------
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=g_dp, minDist=g_minDist, param1=g_param1, param2=g_param2, minRadius=g_minRadius, maxRadius=g_maxRadius)
circles = np.round(circles[0, :]).astype(int)

for (x, y, r) in circles:
    cv2.circle(img_global_circle_detection, (x, y), r, (0, 0, 255), 2)
# cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.imshow('gloval detection', img_global_circle_detection)


#Local Circle Detection ----------------------------------------------------------------------------------------
new_circle = []

# Ensure circles were detected
if circles is not None:
    for (x, y, r) in circles:
        
        if (y - (r+margin) > 0) and (y + (r+margin) < img.shape[0]) and (x - (r+margin) > 0) and (x + (r+margin) <img.shape[1]): # check border line
            roi = img[y - (r+margin): y + (r+margin),x - (r+margin): x + (r+margin)]
        else:
            continue
        
        c = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=l_dp, minDist=l_minDist, param1=l_param1, param2=l_param2, minRadius= r - margin, maxRadius= r + margin)
        if c is not None:
            # roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            # cv2.circle(roi, (round(c[0][0][0]),round(c[0][0][1])), round(c[0][0][2]), (0, 0, 255), 1)
            # cv2.imshow('roi', roi)
            # cv2.waitKey()
            new_circle.append(np.array([c[0][0][0] + x - (r+margin), c[0][0][1] + y - (r+margin), c[0][0][2]]))
    
    for (x, y, r) in new_circle:
        cv2.circle(img_ori, (round(x), round(y)), round(r), (0, 0, 255), 2)
    # cv2.namedWindow('Detected Circles', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Circles', img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles were detected.")

# solvePnp----------------------------------------------------------------------------

PATTERN_SIZE = (8,8) # 18 circle exist
UNIT_SIZE = 15.9375 # distance between circles // unit is millimeter
H, W = img.shape[:2] 


points2Ds = [np.array([[[i[0], i[1]]] for i in new_circle], np.float32)]
print(points2Ds)


pattern_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points *= UNIT_SIZE
points3Ds = [pattern_points]


rms_err, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points3Ds, points2Ds, (W, H), None, None)   
print("\nRMS:", rms_err)
print("camera intrinsic matrix:\n", mtx)      # 카메라 내부 매트릭스
print("distortion coefficients: ", dist.ravel()) # 왜곡 계수 출력


