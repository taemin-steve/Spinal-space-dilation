import cv2
#------------------Parameter Define------------------------------------------------------------------------
# g_* >> variable for global Circle Detection 
# l_* >> variable for local Circle Detection

# dp : This is the inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1, the accumulator has the same resolution as the input image. If dp=2, the accumulator has half the resolution of the input image. It is typically set to a value between 1 and 2.
# minDist: This is the minimum distance between the centers of the detected circles. It specifies the minimum distance (in pixels) between the centers of two circles. If the distance between two circles is less than this value, then the weaker circle is discarded.
# param1: This is the higher threshold of the two passed to the Canny edge detector (the lower threshold is twice smaller). The Hough Circle Transform first applies the Canny edge detector to find potential circle edges in the image. param1 is the higher threshold used by the Canny edge detector.
# param2: This is the accumulator threshold for circle detection. It is a threshold value for the accumulator. The smaller it is, the more false circles may be detected. The larger it is, the more circles need to be found to be considered valid circles.
# minRadius: This is the minimum radius of the circles you want to detect (in pixels).
# maxRadius: This is the maximum radius of the circles you want to detect (in pixels).
#-----------------------------------------------------------------------------------------------------------

# Create a FileStorage object and open a file for writing
fs = cv2.FileStorage("data.yml", cv2.FILE_STORAGE_WRITE)

# Global Circle Detection Variables
fs.write("g_dp", 1)
fs.write("g_minDist", 50)
fs.write("g_param1", 350)
fs.write("g_param2", 15)
fs.write("g_minRadius", 20)
fs.write("g_maxRadius", 40)

# Local Circle Detection Variables dp=1, minDist=50, param1=350, param2=20
fs.write("l_dp", 1)
fs.write("l_minDist", 50)
fs.write("l_param1", 350)
fs.write("l_param2", 20)
fs.write("margin", 10)

# Close the file
fs.release()


fs = cv2.FileStorage("data.yml", cv2.FILE_STORAGE_READ)

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

# Print the read variables
print("Variable1:", l_param2)
print("Variable2:", g_maxRadius)