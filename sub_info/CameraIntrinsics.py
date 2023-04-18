'''
이것은 카메라 내부 파라미터를 구하기 위한 코드입니다.
RealSense 카메라를 노트북에 연결합니다.
연결확인 후, 이 코드를 실행시켜서 내부 파라미터를 구할 수 있습니다.
'''
import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()

# Create a configuration for the pipeline
config = rs.config()

# Enable the color stream with the desired settings
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
profile = pipeline.start(config)

# Get the intrinsics of the color camera
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Print the intrinsics
print("Color Camera Intrinsics:")
print("Width:", intrinsics.width)
print("Height:", intrinsics.height)
print("FX:", intrinsics.fx)
print("FY:", intrinsics.fy)
print("CX:", intrinsics.ppx)
print("CY:", intrinsics.ppy)