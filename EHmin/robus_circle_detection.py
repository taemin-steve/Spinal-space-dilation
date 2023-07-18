import cv2
import numpy as np


def calculate_variance(data):
    data = [i[2] for i in data]
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return variance

# # Apply Hough Circle Transform
img_path= "./c-arm 2023-05-09/7818.png"
img_path= "./c-arm 2023-05-09/7819.png"


img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_ori = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_sel = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


circles = cv2.HoughCircles(img_ori, cv2.HOUGH_GRADIENT_ALT, dp=0.5, minDist=50, param1=300, param2=0.5, minRadius=10, maxRadius=100)

circles = np.round(circles[0, :]).astype(int)

for (x, y, r) in circles:
    cv2.circle(img_ori, (x, y), r, (255, 255, 255), 1)
cv2.imshow('ori', img_ori)

circles = sorted(circles, key=lambda x : x[2])

r_variance = []

for i in range(len(circles) - 7 + 1):
    r_variance.append(calculate_variance(circles[i:i+7]))
    
smallest_index = r_variance.index(min(r_variance))

circles = circles[smallest_index : smallest_index + 7]


for (x, y, r) in circles:
    cv2.circle(img_sel, (x, y), r, (255, 255, 255), 1)
cv2.imshow('select_circle', img_sel)


new_circle = []
# Ensure circles were detected
if circles is not None:
    for (x, y, r) in circles:
        roi = img[y - (r+5): y + (r+5),x - (r+5): x + (r+5) ]
        c = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT_ALT, dp=0.5, minDist=50, param1=500, param2=0.9, minRadius= r-3, maxRadius= r + 3)
        print(c[0][0])
        cv2.circle(roi, (round(c[0][0][0]),round(c[0][0][1])), round(c[0][0][2]), (255, 255, 255), 1)
        cv2.imshow('roi', roi)
        cv2.waitKey()
        new_circle.append(np.array([c[0][0][0] + x - (r+5), c[0][0][1] + y - (r+5), c[0][0][2]]))
        # new_circle.append()
        
    print(new_circle)
    # for (x, y, r) in circles:
    #     cv2.circle(img, (round(x), round(y)), round(r), (0, 255, 0), 2)
        
    cv2.imshow('Detected Circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# else:
    print("No circles were detected.")
    
    


