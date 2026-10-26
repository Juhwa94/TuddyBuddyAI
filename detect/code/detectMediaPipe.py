# mediapipe를 활용하여 얼굴 인식

import cv2
import mediapipe as mp
import os

data_path = "./data"
model_path = "./models/face_landmarker.task"

os.makedirs(data_path, exist_ok=True)

# MediaPipe 설정
BaseOptions = mp.tasks.BaseOptions

# FaceLandmarker 객체를 사용하여 이미지에서 얼굴을 찾아 눈·코·입·윤곽 등의 랜드마크 좌표를 추출
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

ret, frame = cap.read()

if not ret:
    print("프레임을 읽을 수 없습니다.")
    cap.release()
    exit()

# OpenCV BGR → RGB
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# NumPy 이미지 → MediaPipe 이미지
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
)

with FaceLandmarker.create_from_options(options) as landmarker:

    # 얼굴 랜드마크 탐지
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        print("얼굴 감지 성공")

        landmarks = result.face_landmarks[0]

        print("랜드마크 개수:", len(landmarks))

        height, width, _ = frame.shape

        # 랜드마크를 OpenCV 좌표로 변환하여 표시
        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imwrite(f"{data_path}/mediapipe_face.jpg", frame)
        print("랜드마크 이미지 저장 완료")

    else:
        print("얼굴을 감지하지 못했습니다.")

cap.release()