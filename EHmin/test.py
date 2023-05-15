import numpy as np
import cv2 as cv
import math
from PIL import Image

fs = cv.FileStorage("./EHmin/xml/" + str(7822) +'.txt', cv.FILE_STORAGE_READ) 
node = fs.getNode("my_data")
a = node.mat()
print(np.array(a))
fs.release()