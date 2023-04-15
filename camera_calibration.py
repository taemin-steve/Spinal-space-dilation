import cv2 as cv
import numpy as np
from circle import make_2D_points

patternSize = (6,3) # 18 circle exist
unitSize = 47.8125 # distance between circles // unit is millimeter
imgInit = cv.imread('./circle_images/1_Color.png')
h, w = imgInit.shape[:2] 


pattern_points = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
pattern_points[:, :2] = np.indices(patternSize).T.reshape(-1, 2)
pattern_points *= unitSize

# print( make_2D_points().shape)
# print(make_2D_points())

points2Ds = make_2D_points() # make_2D_points return (10,18,2), float32 np array
points3Ds = [pattern_points for i in range(10)]

print(points2Ds)




rms_err, intrisic_mtx, dist_coefs, rvecs, tvecs = cv.calibrateCamera(points3Ds, points2Ds, (w, h), None, None)   
print("\nRMS:", rms_err)
print("camera intrinsic matrix:\n", intrisic_mtx)
print("distortion coefficients: ", dist_coefs.ravel())


