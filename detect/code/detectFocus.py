# detectMediaPipe.py + detectMotion.py

import cv2
import mediapipe as mp
import os
import time

data_path = "./data"
model_path = "./models/face_landmarker.task"

os.makedirs(data_path, exist_ok=True)

# MediaPipe 설정
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# OpenCV 웹캠 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 움직임 비교용 첫 프레임
ret, previous_frame = cap.read()

if not ret:
    print("프레임을 읽을 수 없습니다.")
    cap.release()
    exit()

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# 학습 중 아님 연속 횟수
not_studying_count = 0

print("학습 집중 상태 감지를 시작합니다.")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # =========================
        # 1. 얼굴 감지
        # =========================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)
        face_detected = bool(result.face_landmarks)

        # =========================
        # 2. 움직임 감지
        # =========================
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(previous_gray, current_gray)
        _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        motion_pixels = cv2.countNonZero(threshold)
        motion_detected = motion_pixels > 5000

        # =========================
        # 3. 학습 상태 판단
        # =========================
        studying = face_detected or motion_detected

        if studying:
            not_studying_count = 0
        else:
            not_studying_count += 1

        print(
            f"얼굴: {face_detected} | "
            f"움직임: {motion_detected} ({motion_pixels}) | "
            f"학습 중: {studying} | "
            f"미감지: {not_studying_count}/10"
        )

        # 학습 중 아님이 연속 10번이면 종료
        if not_studying_count >= 10:
            print("학습 상태가 10회 연속 감지되지 않아 종료합니다.")
            break

        # 현재 프레임을 다음 비교 기준으로 변경
        previous_gray = current_gray

        time.sleep(1)

cap.release()

print("학습 집중 상태 감지를 종료합니다.")
