import cv2
import numpy as np


def calculate_variance(data):
    data = [i[2] for i in data]
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return variance

# # Apply Hough Circle Transform
img_path= "./newData/7065.png"
# img_path= "./c-arm 2023-05-09/7819.png"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_ori = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Image that 
# img_sel = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# img_sel = cv2.cvtColor(img_sel, cv2.COLOR_GRAY2BGR)


img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # for houghCircle

img_ori = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # save original image, draw final circle in here
img_ori = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2BGR)

img_global_circle_detection = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # for draw global circle detection
img_global_circle_detection = cv2.cvtColor(img_global_circle_detection, cv2.COLOR_GRAY2BGR)

# -----------------------------------------------------------------------------
_, img_OTSU = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.namedWindow('n', cv2.WINDOW_NORMAL)
cv2.imshow('OTSI', img_OTSU)



circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=350, param2=15, minRadius=20, maxRadius=40)

circles = np.round(circles[0, :]).astype(int)

for (x, y, r) in circles:
    cv2.circle(img_global_circle_detection, (x, y), r, (0, 0, 255), 2)

print(circles)

# cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.imshow('ori', img_global_circle_detection)



# circles = sorted(circles, key=lambda x : x[2])

# r_variance = []

# for i in range(len(circles) - 64 + 1):
#     r_variance.append(calculate_variance(circles[i:i+7]))
    
# smallest_index = r_variance.index(min(r_variance))

# circles = circles[smallest_index : smallest_index + 7]


# for (x, y, r) in circles:
#     cv2.circle(img_sel, (x, y), r, (0, 0, 255), 2)
# cv2.namedWindow('select_circle', cv2.WINDOW_NORMAL)
# cv2.imshow('select_circle', img_sel)


new_circle = []
# Ensure circles were detected
if circles is not None:
    for (x, y, r) in circles:
        roi = img[y - (r+10): y + (r+10),x - (r+10): x + (r+10) ]
        c = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=350, param2=15, minRadius=20, maxRadius=40)
        if c is not None:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            cv2.circle(roi, (round(c[0][0][0]),round(c[0][0][1])), round(c[0][0][2]), (0, 0, 255), 1)
            cv2.imshow('roi', roi)
            cv2.waitKey()
            new_circle.append(np.array([c[0][0][0] + x - (r+5), c[0][0][1] + y - (r+5), c[0][0][2]]))
        
    print(new_circle)
    for (x, y, r) in circles:
        cv2.circle(img_ori, (round(x), round(y)), round(r), (0, 0, 255), 2)
    # cv2.namedWindow('Detected Circles', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Circles', img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# else:
    print("No circles were detected.")
    
    


