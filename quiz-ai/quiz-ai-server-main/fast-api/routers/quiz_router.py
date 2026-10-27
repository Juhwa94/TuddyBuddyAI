import asyncio
import hmac
import json
import logging

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from starlette.concurrency import run_in_threadpool
from common import MAX_IMAGE_BYTES

from schemas.quiz import Difficulty, QuizResponse, QuizType
from services.image_service import normalize_image
from services.pipeline_service import run_pipeline
from services.study_service import extract_study_material
from services.taxonomy_service import classify_study_material

# MAX_IMAGE_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ai",
    tags=["quiz"],
)


@router.post("/classify", response_model=QuizResponse)
async def classify_quiz(
    request: Request,
    file: UploadFile = File(...),
    subject_id: int | None = Form(default=None, gt=0),
    catalog_json: str = Form("[]", max_length=20000),
    x_ai_token: str | None = Header(default=None),
):
    settings = request.app.state.settings
    if not settings.token:
        raise HTTPException(status_code=503, detail="AI 서버 인증 토큰이 설정되지 않았습니다.")
    if not hmac.compare_digest(settings.token.encode(), (x_ai_token or "").encode()):
        raise HTTPException(status_code=401, detail="AI 서버 인증 토큰이 올바르지 않습니다.")

    try:
        catalog = json.loads(catalog_json)
        if not isinstance(catalog, list):
            raise ValueError("분류 목록 형식이 올바르지 않습니다.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="분류 목록을 읽지 못했습니다.") from exc

    try:
        data = await file.read(MAX_IMAGE_BYTES + 1)
        declared_mime = file.content_type or ""
    finally:
        await file.close()
    if not data:
        raise HTTPException(status_code=400, detail="이미지가 비어 있습니다.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="10MB 이하 이미지를 선택해주세요.")
    try:
        data, mime = await run_in_threadpool(normalize_image, data, declared_mime)
        async with asyncio.timeout(160):
            material = await extract_study_material(request.app.state.ai, settings.model, data, mime)
            if not material.chunks:
                raise ValueError("학습 자료를 추출하지 못했습니다.")
            taxonomy = await classify_study_material(
                request.app.state.ai, settings.model, material, catalog, subject_id
            )
        return QuizResponse(
            categoryName=taxonomy.categoryName,
            subjectName=taxonomy.subjectName,
            quizzes=[],
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="AI 분류 시간이 초과되었습니다.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="이미지에서 카테고리와 과목을 분류하지 못했습니다.") from exc


@router.post(
    "/generate-quiz",
    response_model=QuizResponse,
)
async def generate_quiz(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    subject_id: int | None = Form(default=None, gt=0),
    catalog_json: str = Form("[]", max_length=20000),
    quiz_type: QuizType = Form(...),
    quiz_difficulty: Difficulty = Form(...),
    quiz_count: int = Form(..., ge=1, le=10),
    quiz_prompt: str = Form("", max_length=1000),
    category_name: str | None = Form(default=None, max_length=100),
    subject_name: str | None = Form(default=None, max_length=100),
    x_ai_token: str | None = Header(default=None),
):
    settings = request.app.state.settings

    try:
        catalog = json.loads(catalog_json)
        if not isinstance(catalog, list):
            raise ValueError("기존 분류 목록 형식이 올바르지 않습니다.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="기존 분류 목록을 읽지 못했습니다.",
        ) from exc

    # 설정 누락 시에도 인증을 우회하지 않도록 fail-closed 처리
    if not settings.token:
        raise HTTPException(
            status_code=503,
            detail="AI 서버 인증 토큰이 설정되지 않았습니다.",
        )

    received_token = (x_ai_token or "").encode()
    expected_token = settings.token.encode()

    if not hmac.compare_digest(expected_token, received_token):
        response.headers["X-AI-Verification"] = "AUTH_FAILED"

        raise HTTPException(
            status_code=401,
            detail="AI 서버 인증에 실패했습니다.",
        )

    # 업로드 파일 읽기
    try:
        data = await file.read(MAX_IMAGE_BYTES + 1)
        declared_mime = file.content_type or ""
    finally:
        await file.close()

    if not data:
        response.headers["X-AI-Verification"] = "IMAGE_EMPTY"

        raise HTTPException(
            status_code=400,
            detail="이미지가 비어 있습니다.",
        )

    if len(data) > MAX_IMAGE_BYTES:
        response.headers["X-AI-Verification"] = "IMAGE_TOO_LARGE"

        raise HTTPException(
            status_code=413,
            detail="10MB 이하 이미지를 선택해주세요.",
        )

    # 이미지 형식과 크기 검증
    try:
        data, mime = await run_in_threadpool(
            normalize_image,
            data,
            declared_mime,
        )
    except ValueError as exc:
        response.headers["X-AI-Verification"] = "IMAGE_INVALID"

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # 이미지 추출 → 문제 생성 → 정답 검증 → 해설 검증
    try:
        # Spring 기본 read timeout 180초보다 짧게 제한
        async with asyncio.timeout(160):
            result = await run_pipeline(
                ai=request.app.state.ai,
                model=settings.model,
                data=data,
                mime=mime,
                subject_id=subject_id,
                quiz_type=quiz_type,
                difficulty=quiz_difficulty,
                count=quiz_count,
                extra=quiz_prompt,
                catalog=catalog,
                category_name=category_name,
                subject_name=subject_name,
            )

        # 검증을 모두 통과한 경우
        response.headers["X-AI-Verification"] = "PASSED"
        response.headers["X-AI-Quiz-Count"] = str(len(result.quizzes))

        logger.info(
            "AI 검증 완료: status=PASSED, quiz_count=%d, quiz_type=%s",
            len(result.quizzes),
            quiz_type,
        )

        return result

    except TimeoutError as exc:
        response.headers["X-AI-Verification"] = "TIMEOUT"

        logger.warning(
            "AI 처리 시간 초과: quiz_type=%s, quiz_count=%d",
            quiz_type,
            quiz_count,
        )

        raise HTTPException(
            status_code=504,
            detail="AI 생성 시간이 초과되었습니다. 다시 시도해주세요.",
        ) from exc

    except ValueError as exc:
        response.headers["X-AI-Verification"] = "FAILED"

        logger.exception(
            "AI 검증 실패: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "자료가 부족하거나 AI 응답 검증에 실패했습니다. "
                "다른 이미지로 다시 시도해주세요."
            ),
        ) from exc

    except Exception as exc:
        response.headers["X-AI-Verification"] = "ERROR"

        logger.exception(
            "AI 서버 요청 실패: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "AI 연결, 모델, API 키 또는 "
                "사용 한도를 확인해주세요."
            ),
        ) from exc
