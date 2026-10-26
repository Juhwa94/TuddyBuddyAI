import cv2
import os

data_path = "./data"
os.makedirs(data_path, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 첫 번째 프레임
ret, frame1 = cap.read()

if not ret:
    print("첫 번째 프레임을 읽을 수 없습니다.")
    cap.release()
    exit()

# 비교하기 쉽도록 흑백 변환
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

print("움직임 감지를 시작합니다.")

while True:
    # 현재 프레임
    ret, frame2 = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 이전 프레임과 현재 프레임의 차이
    diff = cv2.absdiff(gray1, gray2)

    # 차이가 일정 값 이상인 픽셀만 흰색으로 변환
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 움직임이 발생한 픽셀 개수
    motion_pixels = cv2.countNonZero(threshold)

    if motion_pixels > 5000:
        print("움직임 감지:", motion_pixels)
        cv2.imwrite(f"{data_path}/motion.jpg", frame2)
        break

    # 현재 프레임을 다음 비교의 이전 프레임으로 사용
    gray1 = gray2

cap.release()

print("움직임 감지 종료")