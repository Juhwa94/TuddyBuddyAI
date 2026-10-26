# TuddyBuddyAI

TuddyBuddy 프로젝트에서 사용하는 AI 모듈의 기능과 연동 방식을 정리합니다.

각 모듈은 하나의 2단계 제목(`## 모듈명 — 기능명`) 아래 **개요 → 사용 기술 → 주요 기능 → 처리 흐름 → API → 입출력 → 구성 파일 → 실행 및 연동 → 제약 사항** 순서로 작성합니다. 해당하지 않는 항목은 `해당 없음`으로 표시합니다.

## detect — 학습 상태 감지

### 개요

웹캠 이미지를 받아 얼굴 유무와 프레임 간 움직임을 분석하고, 학습 상태 및 타이머 일시정지 판단에 사용할 신호를 반환합니다.

### 사용 기술

| 구분 | 기술 및 역할 |
| --- | --- |
| 얼굴 감지 모델 | MediaPipe FaceLandmarker (`face_landmarker.task`), 이미지당 얼굴 최대 1개 감지 |
| 이미지 처리 | OpenCV, NumPy: 이미지 디코딩·크기 조정·프레임 차이 계산 |
| API 서버 | FastAPI: 이미지 업로드 및 감지 세션 관리 |
| 서버 실행 | Uvicorn |

### 주요 기능

- **얼굴 감지**: 얼굴 랜드마크 검출 여부를 얼굴 존재 여부로 반환합니다.
- **움직임 감지**: 이전 프레임과 현재 프레임의 회색조 차이를 비교합니다. 밝기 차이가 30을 초과하는 픽셀이 5,000개를 초과하면 움직임으로 판단합니다.
- **학습 상태 판단**: 얼굴 또는 움직임 중 하나라도 감지되면 `studying=true`를 반환합니다.
- **연속 미감지 추적**: 얼굴과 움직임이 모두 없으면 연속 미감지 횟수와 시간을 기록합니다. 감지가 재개되면 두 값을 초기화합니다.
- **일시정지 신호**: 연속 미감지 시간이 10초 이상이면 `stopped=true`를 반환합니다. 감지 API와 세션은 계속 유지되어 복귀를 감지할 수 있습니다.
- **세션별 상태 관리**: 이전 프레임과 미감지 기록을 세션별로 분리합니다. 최대 100개 세션을 유지하며, 300초 이상 요청이 없는 세션은 세션 생성 또는 조회 시 정리합니다.

### 처리 흐름

1. 클라이언트가 감지 세션을 생성하고 `session_id`를 받습니다.
2. 웹캠 이미지를 `X-Detect-Session` 헤더와 함께 집중 감지 API로 전송합니다.
3. 서버가 이미지를 640 × 480으로 조정하고 얼굴 감지 및 이전 프레임과의 움직임 비교를 수행합니다.
4. 학습 상태와 연속 미감지 시간을 계산하여 응답합니다.
5. 클라이언트가 응답에 따라 학습 상태를 표시하고 타이머를 제어합니다. 감지 종료 시 세션 삭제를 요청합니다.

### API

| 메서드 | 경로 | 기능 | 요청 |
| --- | --- | --- | --- |
| POST | `/ai/detect/sessions` | 세션 생성, HTTP 201 반환 | 본문 없음 |
| DELETE | `/ai/detect/sessions` | 세션 종료 | `X-Detect-Session` 헤더 |
| POST | `/ai/detect/face` | 단일 이미지 얼굴 감지 | multipart/form-data의 `file` 필드, 세션 불필요 |
| POST | `/ai/detect/focus` | 얼굴·움직임 기반 학습 상태 감지 | multipart/form-data의 `file` 필드와 `X-Detect-Session` 헤더 |

### 입출력

**입력**: OpenCV로 디코딩 가능한 이미지 파일, 최대 5MiB(5 × 1024 × 1024바이트). 집중 감지는 먼저 생성한 세션 ID가 필요합니다.

**출력**: 세션 생성은 `success`, `session_id`, 세션 종료는 `success`, 얼굴 감지는 `success`, `face_detected`를 반환합니다. 집중 감지 응답 필드는 다음과 같습니다.

| 필드 | 타입 | 의미 |
| --- | --- | --- |
| `success` | boolean | 요청 처리 성공 여부 |
| `face_detected` | boolean | 얼굴 감지 여부 |
| `motion_detected` | boolean | 움직임 감지 여부 |
| `motion_pixels` | integer | 밝기 차이 임계값을 초과한 픽셀 수 |
| `studying` | boolean | 얼굴 또는 움직임 감지 여부 |
| `not_studying_count` | integer | 연속 미감지 요청 횟수 |
| `not_studying_seconds` | number | 최초 미감지 시점부터 경과한 시간(초), 소수 둘째 자리로 반올림 |
| `stopped` | boolean | 연속 미감지 10초 이상일 때 타이머 일시정지 판단 신호 |

오류는 HTTP 상태 코드와 `detail`로 반환합니다. 대표적으로 잘못된 이미지·필수 세션 헤더 누락은 400, 없거나 만료된 세션은 404, 이미지 용량 초과는 413, 세션 수 초과·모델 미준비는 503입니다.

### 구성 파일

| 경로 | 역할 |
| --- | --- |
| `detect/body/detectAPI.py` | 서비스용 FastAPI 서버 및 감지 로직 |
| `detect/body/models/face_landmarker.task` | 서비스용 얼굴 랜드마크 모델 |
| `detect/body/requirements/requirements.txt` | 원본 실행 환경의 패키지 목록 |
| `detect/code/detectCV.py` | 웹캠 촬영 예제 |
| `detect/code/detectMediaPipe.py` | 얼굴 랜드마크 감지 예제 |
| `detect/code/detectMotion.py` | 움직임 감지 예제 |
| `detect/code/detectFocus.py` | 얼굴·움직임 통합 감지 예제 |
| `detect/code/models/face_landmarker.task` | 예제용 모델 |
| `detect/requirements/requirements.txt` | 원본 패키지 목록 사본 |

### 실행 및 연동

필요한 주요 패키지는 `mediapipe`, `opencv-python`, `numpy`, `fastapi`, `python-multipart`, `uvicorn`입니다. 기존 requirements에는 다른 기능의 패키지와 원본 환경의 로컬 파일 경로가 포함되어 있어 환경에 맞는 의존성 정리가 필요합니다.

의존성을 준비한 후 저장소 루트에서 실행합니다.

```powershell
cd detect/body
python -m uvicorn detectAPI:app --host 0.0.0.0 --port 8000
```

TuddyBuddy 프론트의 `REACT_APP_DETECT_API_URL`에 서버 주소를 설정합니다. 기존 프론트는 640 × 480 JPEG 이미지를 전송하고, 응답을 받은 뒤 약 1초 후 다음 감지를 요청합니다. 현재 서버의 CORS 허용 출처는 `http://localhost:3000`입니다. 다른 출처에서 연동하려면 서버 설정을 수정해야 합니다.

### 제약 사항

- 학습 상태는 얼굴·움직임 유무를 기준으로 추정합니다. 시선, 졸음, 실제 학습 내용이나 집중도를 직접 판별하지 않습니다.
- 움직임은 화면 전체를 비교하므로 배경 변화나 조명 변화도 학습 중 판정에 영향을 줄 수 있습니다. 첫 프레임에는 비교 대상이 없어 움직임을 감지하지 않습니다.
- `stopped`는 응답 신호이며 실제 타이머 일시정지와 재개는 클라이언트가 처리해야 합니다.
- 세션은 프로세스 메모리에 저장되므로 서버 재시작 시 사라집니다. 여러 워커나 서버를 사용할 경우 세션 공유 또는 동일 프로세스로의 요청 라우팅이 필요합니다.
- `detect/code`의 파일은 독립 실행 예제입니다. 특히 `detectFocus.py`는 연속 10회 미감지 시 종료하며, 서비스 API의 시간 기반 판정과 동작이 다릅니다.
- 이 저장소로 복사한 뒤 서버 실행 및 프론트 연동은 아직 검증하지 않았습니다.
