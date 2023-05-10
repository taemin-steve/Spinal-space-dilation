import cv2

# FileStorage 객체 생성
fs = cv2.FileStorage('example.xml', cv2.FILE_STORAGE_WRITE)

# 파일에 데이터 쓰기
fs.write("my_data", str([1, 2, 3, 4, 5]))

# FileStorage 객체 해제
fs.release()
