# 카메라 연결 확인

import cv2
import os

data_path = "./data"
os.makedirs(data_path, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

ret, frame = cap.read()

if ret:
    print("카메라 연결 성공")
    print("frame shape:", frame.shape)

    cv2.imwrite(f"{data_path}/camera_test.jpg", frame)
    print("이미지 저장 완료")
else:
    print("프레임을 읽을 수 없습니다.")

cap.release()