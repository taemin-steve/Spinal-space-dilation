import math
import numpy as np

unit = 15.9375
unit_position = [3,3]

marker_r = 12.7 / 2
plane_r = 3 # 홈의 반지름
plane_h = 5 # 기둥 높이 

# h = math.sqrt(marker_r**2 - plane_r**2) + plane_h

small_marker_r = 14/2 #작은 마커 반지름
small_h = small_marker_r - math.sqrt(small_marker_r**2 - plane_r**2) #빠진 길이

big_marker_r = 15.9/2 # 큰 마커 반지름 
big_h = big_marker_r - math.sqrt(big_marker_r**2 - plane_r**2)

h = big_h - small_h

# x_prime = math.sqrt(49 - 9)
# h = (5 + 7 + x_prime) - (5 + math.sqrt(small_marker_r**2 - plane_r**2))
# h = small_marker_r 
# h = (5 + math.sqrt(small_marker_r**2 - plane_r**2)) - (5 + x_prime + 4)


point1 = [-222.377,-251.387,1540.890]
point2 = [-129.497,-267.920,1480.996]
point3 = [-284.297,-277.899,1452.375]
point4 = [-191.275,-294.987,1392.297]

points = [point1, point2,point3,point4]

real_marker = [-203.597,-263.655,1474.820]

#check
print(unit * 7)
print(np.linalg.norm(np.array(point2) - np.array(point1)))
print(np.linalg.norm(np.array(point3) - np.array(point1)))
print()

def calc_normal_vector(point1, point2, point3):
    vector_x = np.array(point2) - np.array(point1)
    vector_y = np.array(point3) - np.array(point1)
    
    normal_vector = np.cross(vector_x, vector_y)
    
    vector_x /= np.linalg.norm(vector_x)
    vector_y /= np.linalg.norm(vector_y)
    normal_vector /= np.linalg.norm(normal_vector)
    
    return vector_x, vector_y, normal_vector

def fit_plane_to_points(points):
    # Convert points to NumPy array for easier manipulation
    points = np.array(points)

    # Center the points by subtracting their mean
    centered_points = points - np.mean(points, axis=0)

    # Perform Singular Value Decomposition (SVD) to find the normal vector of the best-fitting plane
    _, _, V = np.linalg.svd(centered_points)

    # The normal vector is the last row of V
    normal_vector = V[-1]

    # Normalize the normal vector to have unit length
    normal_vector /= np.linalg.norm(normal_vector)

    return normal_vector


x, y, z = calc_normal_vector(point1, point2, point3)
z = fit_plane_to_points(points)
# print(x,y,z)

estimate_marker = point1 + unit_position[0]*unit*(x) + unit_position[1]*unit*(y) #+ h*(z)

print(real_marker)
print(estimate_marker)

print(np.linalg.norm(np.array(estimate_marker) - np.array(real_marker)))

# point1 = [-205.956,-264.292,1474.825]
# point2 = [-209.105,-270.486,1477.136]
# print(np.linalg.norm(np.array(point2) - np.array(point1)))




