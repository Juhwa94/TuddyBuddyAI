from typing import List, Optional, Any, Literal
from pydantic import BaseModel, Field


class CategoryReference(BaseModel):
    categoryIds: List[int] = Field(default_factory=list)
    categoryName: str


class CategoryAnalysisItem(CategoryReference):
    mentorComment: str = Field(min_length=1, max_length=500, description="해당 카테고리의 확인 가능한 취약점 요약")
    weaknesses: List[str] = Field(max_length=10, description="해당 카테고리에서 근거가 확인되는 오답과 원인")
    consultingPath: List[str] = Field(min_length=3, max_length=3, description="해당 카테고리의 3단계 학습 경로")


class LLMFeedbackResponse_Statistics(BaseModel):
    categoryAnalyses: List[CategoryAnalysisItem] = Field(min_length=1)


class QuizLog(BaseModel):
    quiz_id: int
    user_answer: Optional[str] = Field(None, alias="answer")
    is_correct: Optional[Any] = None
    question: Optional[Any] = None
    categoryId: Optional[int] = None
    categoryName: Optional[str] = None
    categoryGroup: Optional[str] = None


class AnalyzeRequest(BaseModel):
    user_id: Optional[int] = None
    memberNum: Optional[int] = None
    quiz_logs: List[QuizLog] = []


class PredictionItem(BaseModel):
    name: Literal["최근 평균", "다음 시험 예측"] = Field(description="항목명")
    score: float = Field(ge=0, le=100, allow_inf_nan=False, description="산출된 점수 (0~100 사이의 숫자)")
    type: Literal["actual", "predict"] = Field(description="구분 타입")


class CategoryPredictionItem(CategoryReference):
    prediction: List[PredictionItem] = Field(min_length=2, max_length=2)
    mentorComment: str = Field(min_length=1, max_length=1000)


class LlmPredictionResponse_Pydantic(BaseModel):
    categoryPredictions: List[CategoryPredictionItem] = Field(min_length=1)


class CategoryStatItem(BaseModel):
    categoryId: Optional[int] = None
    categoryName: Optional[str] = None
    categoryGroup: Optional[str] = None
    totalAttempts: Optional[int] = Field(default=None, ge=0)
    averageScore: Optional[float] = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    maxScore: Optional[float] = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    minScore: Optional[float] = Field(default=None, ge=0, le=100, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    stats: List[CategoryStatItem] = Field(default_factory=list)
