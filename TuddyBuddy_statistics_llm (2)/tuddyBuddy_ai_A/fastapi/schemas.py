import os
import json
from typing import List, Optional, Any
from pydantic import BaseModel, Field

# ==========================================
# 1. Pydantic 스키마 정의 (Schemas)
# ==========================================

class LLMFeedbackResponse_Statistics(BaseModel):
    mentorComment: str = Field(description="전체적인 성적 흐름과 취약점 한 줄 요약")
    weaknesses: List[str] = Field(description="틀린 문제들의 이유를 분석하여 단원명과 구체적 원인 목록")
    consultingPath: List[str] = Field(description="차후 성적 향상을 위한 단계별 순서 목록")

class QuizLog(BaseModel):
    quiz_id: int
    user_answer: Optional[str] = Field(None, alias="answer") # 자바쪽 키값 대응
    is_correct: Optional[Any] = None
    question: Optional[Any] = None # 자바가 보내는 question 필드 수용

class AnalyzeRequest(BaseModel):
    user_id: Optional[int] = None
    memberNum: Optional[int] = None  # 자바에서 혹시 memberNum으로 보낼 경우 대비
    quiz_logs: List[QuizLog] = []

#=========================================
# 2. 점수예측 스키마 정의
#=========================================
# 개별 예측/평균 데이터 항목
class PredictionItem(BaseModel):
    name: str = Field(description="항목명 ('최근 평균', '다음 시험 예측' 등)")
    score: float = Field(description="산출된 점수 (0~100 사이의 숫자)")
    type: str = Field(description="구분 타입 ('actual' 또는 'predict')")

# 스프링 부트로 최종 반환될 LLM 분석/예측 응답 구조
class LlmPredictionResponse_Pydantic(BaseModel):
    prediction: List[PredictionItem] = Field(description="최근 평균 점수와 다음 시험 예측 점수 목록")
    mentorComment: str = Field(description="예측 점수에 대한 AI 멘토의 분석 코멘트")

# 카테고리별 통계 단건 아이템
class CategoryStatItem(BaseModel):
    categoryId: Optional[int] = None
    categoryName: Optional[str] = None
    totalAttempts: Optional[int] = None
    averageScore: Optional[float] = None
    maxScore: Optional[float] = None
    minScore: Optional[float] = None

# /prediction 전용 요청 스키마 (스프링에서 List[Map] 형태로 통째로 보낼 때 대응)
class PredictionRequest(BaseModel):
    stats: List[CategoryStatItem] = []