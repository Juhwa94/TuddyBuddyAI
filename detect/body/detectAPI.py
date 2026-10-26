from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Optional
import secrets
import time

import cv2
import mediapipe as mp
import numpy as np

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool


# =========================
# 설정
# =========================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_landmarker.task"
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MOTION_THRESHOLD = 30
MOTION_PIXEL_THRESHOLD = 5000

# 연속 비학습 시간
NOT_STUDYING_SECONDS = 10.0

# 일정 시간 요청이 없으면 감지 세션 만료
SESSION_TTL_SECONDS = 300.0
MAX_SESSIONS = 100

# 업로드 이미지 최대 크기
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


# =========================
# 감지 세션
# =========================

@dataclass
class DetectionSession:
    previous_gray: Optional[np.ndarray] = None
    not_studying_count: int = 0
    not_studying_since: Optional[float] = None
    last_seen: float = 0.0


sessions: dict[str, DetectionSession] = {}

# 공유 MediaPipe 모델과 세션 상태의 동시 접근 보호
state_lock = RLock()

landmarker = None


def remove_expired_sessions(now: float) -> None:
    """state_lock 내부에서 호출합니다."""
    expired_ids = [
        session_id
        for session_id, session in sessions.items()
        if now - session.last_seen >= SESSION_TTL_SECONDS
    ]

    for session_id in expired_ids:
        sessions.pop(session_id, None)


def require_session(session_id: Optional[str]) -> DetectionSession:
    """state_lock 내부에서 호출합니다."""
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="X-Detect-Session 헤더가 필요합니다."
        )

    now = time.monotonic()
    remove_expired_sessions(now)

    session = sessions.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="감지 세션이 없거나 만료되었습니다."
        )

    return session


# =========================
# 모델 초기화·종료
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global landmarker

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            "models/face_landmarker.task 파일을 확인해주세요."
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1
    )

    landmarker = (
        mp.tasks.vision.FaceLandmarker.create_from_options(options)
    )

    try:
        yield
    finally:
        with state_lock:
            sessions.clear()

            if landmarker is not None:
                landmarker.close()
                landmarker = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# =========================
# 이미지 처리
# =========================

async def read_image(file: UploadFile) -> bytes:
    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="이미지 파일이 비어 있습니다."
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="이미지는 5MB 이하로 전송해주세요."
        )

    return image_bytes


def decode_frame(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="이미지를 읽을 수 없습니다."
        )

    # 프레임 크기를 고정해 움직임 비교 기준을 유지합니다.
    return cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )


def detect_face_in_frame(frame: np.ndarray) -> bool:
    """state_lock 내부에서 호출합니다."""
    if landmarker is None:
        raise HTTPException(
            status_code=503,
            detail="감지 모델이 준비되지 않았습니다."
        )

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    return bool(result.face_landmarks)


# =========================
# 감지 세션 시작
# =========================

@app.post("/ai/detect/sessions", status_code=201)
def create_detection_session():
    now = time.monotonic()

    with state_lock:
        remove_expired_sessions(now)

        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=503,
                detail="감지 세션이 많습니다. 잠시 후 다시 시도해주세요."
            )

        session_id = secrets.token_urlsafe(32)

        while session_id in sessions:
            session_id = secrets.token_urlsafe(32)

        sessions[session_id] = DetectionSession(
            last_seen=now
        )

    return {
        "success": True,
        "session_id": session_id
    }


# =========================
# 감지 세션 종료
# =========================

@app.delete("/ai/detect/sessions")
def close_detection_session(
    session_id: Optional[str] = Header(
        default=None,
        alias="X-Detect-Session"
    )
):
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="X-Detect-Session 헤더가 필요합니다."
        )

    with state_lock:
        # 이미 종료한 세션도 종료 성공으로 처리합니다.
        sessions.pop(session_id, None)

    return {
        "success": True
    }


# =========================
# 얼굴 감지 API
# =========================

def process_face(image_bytes: bytes):
    frame = decode_frame(image_bytes)

    with state_lock:
        face_detected = detect_face_in_frame(frame)

    return {
        "success": True,
        "face_detected": face_detected
    }


@app.post("/ai/detect/face")
async def detect_face(file: UploadFile = File(...)):
    image_bytes = await read_image(file)

    return await run_in_threadpool(
        process_face,
        image_bytes
    )


# =========================
# 학습 집중 감지 API
# =========================

def process_focus(
    image_bytes: bytes,
    session_id: Optional[str]
):
    frame = decode_frame(image_bytes)

    with state_lock:
        session = require_session(session_id)

        # 1. 얼굴 감지
        face_detected = detect_face_in_frame(frame)

        # 2. 움직임 감지
        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        motion_pixels = 0
        motion_detected = False

        if session.previous_gray is not None:
            diff = cv2.absdiff(
                session.previous_gray,
                current_gray
            )

            _, threshold = cv2.threshold(
                diff,
                MOTION_THRESHOLD,
                255,
                cv2.THRESH_BINARY
            )

            motion_pixels = int(
                cv2.countNonZero(threshold)
            )

            motion_detected = (
                motion_pixels > MOTION_PIXEL_THRESHOLD
            )

        session.previous_gray = current_gray

        # 기존 규칙 유지:
        # 얼굴 또는 움직임이 있으면 학습 중으로 판단
        studying = face_detected or motion_detected

        now = time.monotonic()

        if studying:
            session.not_studying_count = 0
            session.not_studying_since = None
            not_studying_seconds = 0.0
        else:
            session.not_studying_count += 1

            if session.not_studying_since is None:
                session.not_studying_since = now

            not_studying_seconds = (
                now - session.not_studying_since
            )

        session.last_seen = now

        # 타이머 일시정지 판단용 신호입니다.
        # 감지 API 자체를 종료하는 값은 아닙니다.
        stopped = (
            not studying
            and not_studying_seconds >= NOT_STUDYING_SECONDS
        )

        return {
            "success": True,
            "face_detected": face_detected,
            "motion_detected": motion_detected,
            "motion_pixels": motion_pixels,
            "studying": studying,
            "not_studying_count": session.not_studying_count,
            "not_studying_seconds": round(
                not_studying_seconds,
                2
            ),
            "stopped": stopped
        }


@app.post("/ai/detect/focus")
async def detect_focus(
    file: UploadFile = File(...),
    session_id: Optional[str] = Header(
        default=None,
        alias="X-Detect-Session"
    )
):
    image_bytes = await read_image(file)

    return await run_in_threadpool(
        process_focus,
        image_bytes,
        session_id
    )