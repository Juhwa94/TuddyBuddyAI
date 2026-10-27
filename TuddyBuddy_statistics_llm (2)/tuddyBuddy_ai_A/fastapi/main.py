import os
import json
from typing import List, Optional, Any
from fastapi import FastAPI, status, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


from schemas import AnalyzeRequest,CategoryStatItem

from service.llm_service import analyze_user_statistics,analyze_score_prediction

app = FastAPI()

# FastAPI 엔드포인트 
@app.post("/statistics", status_code=status.HTTP_200_OK)
async def analyze_performance(request: AnalyzeRequest):
    if not request.quiz_logs:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "mentorComment": "분석할 오답 데이터가 없습니다. 먼저 퀴즈를 풀고 도전해 보세요!",
                "weaknesses": [],
                "consultingPath": ["1. 학습 메뉴로 이동하여 신규 퀴즈를 풀어보세요."]
            }
        )

    try:
        quiz_logs_dict = [log.model_dump() for log in request.quiz_logs]
        feedback = analyze_user_statistics(request.user_id, quiz_logs_dict)
        print("===== [statistics LLM 분석 응답 결과] =====")
        print(feedback)
        print("================================")
        return feedback
        
    except Exception as e:
        import traceback
        traceback.print_exc()  # 콘솔에 상세 트레이스백 출력
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )


@app.post("/prediction", status_code=status.HTTP_200_OK)
async def analyze_prediction_endpoint(stats_list: List[CategoryStatItem]):
    if not stats_list:
        return {
            "prediction": [
                {"name": "최근 평균", "score": 0.0, "type": "actual"},
                {"name": "다음 시험 예측", "score": 0.0, "type": "predict"}
            ],
            "mentorComment": "분석할 카테고리 통계 데이터가 없습니다."
        }

    try:
        # Pydantic 모델 리스트를 dict 리스트로 변환하여 LLM 서비스에 전달
        stats_dict_list = [stat.model_dump() for stat in stats_list]
        
        # 점수 예측 LLM 서비스 호출 (통계 데이터를 기반으로 분석)

        prediction_result = analyze_score_prediction(stats_dict_list)
        print("===== [prediction LLM 분석 응답 결과] =====")
        print(prediction_result)
        print("================================")
        return prediction_result
        
    except Exception as e:
        import traceback
        traceback.print_exc()  # 콘솔에 상세 트레이스백 출력
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("===== [422 Validation Error 상세] =====")
    print(exc.errors())
    print("========================================")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )



"""
uvicorn main:app --host localhost --port 8001 --reload

analyze_prediction_endpoint 더미데이터 출력결과
{
  "mentorComment": "답안을 제출하지 못한 문항들이 확인되어 기본 개념에 대한 이해와 실전 시간 배분 연습이 시급합니다.",
  "weaknesses": [
    "기초 개념 단원(101번 문항): 개념 이해 부족으로 인한 답안 미제출",
    "응용 유형 단원(102번 문항): 풀이 시간 부족 및 문제 접근 전략 미흡으로 인한 무응답"
  ],
  "consultingPath": [
    "1단계: 101번, 102번 관련 단원의 핵심 이론 개념 재정립",
    "2단계: 유형별 대표 문항 풀이를 통한 단계별 접근법 연습",
    "3단계: 제한시간을 두고 푸는 실전 연습을 통해 시간 관리 능력 강화"
  ]
}
"""

"""

validation_exception_handler 더미데이터 출력결과
{
  "prediction": [
    {
      "name": "최근 평균",
      "score": 78.3,
      "type": "actual"
    },
    {
      "name": "다음 시험 예측",
      "score": 81.5,
      "type": "predict"
    }
  ],
  "mentorComment": "전체 학습 데이터를 분석한 결과 최근 평균 점수는 78.3점입니다. 자바 기초 영역은 평균 82.5점으로 안정적인 실력을 유지하고 있으나, 데이터베이스/JDBC 영역은 71.5점으로 상대적으로 약세를 보이고 있습니다. 데이터베이스 관련 개념과 오답을 집중 보완한다면 다음 시험에서 81.5점 수준의 성적을 거둘 수 있을 것으로 예상됩니다."
}
"""