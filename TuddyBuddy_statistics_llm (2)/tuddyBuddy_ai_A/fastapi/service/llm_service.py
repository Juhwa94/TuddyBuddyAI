import os
import json
from typing import List, Optional, Any
from fastapi import FastAPI, status, HTTPException, Request

# 경로 설정용
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

# common.py에서 필요한 설정값 가져오기
from common import GEMINI_MODEL, get_chat

from schemas import LLMFeedbackResponse_Statistics,LlmPredictionResponse_Pydantic



def analyze_user_statistics(user_id: int, quiz_logs: list) -> dict:
    llm = get_chat(provider="gemini")
    
    prompt = f"""
    당신은 친절하면서도 명확하게 핵심을 짚어주는 AI 멘토입니다.
    아래 유저의 오답 문제 내역 및 해설 데이터를 바탕으로 분석 피드백을 작성하세요.

    [유저 오답 데이터]
    {json.dumps(quiz_logs, ensure_ascii=False, indent=2)}

    다음 요구사항을 지켜주세요:
    1. mentorComment에는 전체적인 성적 흐름과 취약점 한 줄 요약을 작성 할 것
    2. weaknesses에는 틀린 문제들의 이유를 분석하여 단원명과 구체적 원인을 적을 것
    3. consultingPath에는 차후 성적 향상을 위한 단계별 순서(1, 2, 3...)를 3단계로 제시 할 것
    """
    
    structured_llm = llm.with_structured_output(LLMFeedbackResponse_Statistics)
    response = structured_llm.invoke(prompt)
    return response


def analyze_score_prediction( quiz_logs: list) -> LlmPredictionResponse_Pydantic:
    llm = get_chat(provider="gemini", temperature=0.2)
    
    prompt = f"""
    당신은 전문 교육 데이터 분석가이자 AI 멘토입니다.
    아래 유저의 퀴즈 및 오답 로그 데이터를 바탕으로 유저의 '최근 평균' 점수를 객관적으로 산출하고, 이를 바탕으로 '다음 시험 예측' 점수를 분석하여 예측하세요.

    1. 코드가 입력 통계로 평균·정답률 등 계산 가능한 값을 직접 산출 할 것
    2. LLM은 계산 결과와 로그를 바탕으로 설명과 학습 조언을 생성 할 것
    3. 코드가 반환 결과의 점수·항목 수·범위를 검증하고, 설명에 입력에 없는 수치나 근거 없는 단정이 있는지 제한적으로 확인 할 것

    [유저 학습/오답 데이터]
    {json.dumps(quiz_logs, ensure_ascii=False, indent=2)}

    다음 요구사항을 반드시 지켜주세요:
    1. prediction 배열에는 정확히 2개의 아이템을 포함 할 것
       - 첫 번째 아이템: name은 '최근 평균', score는 데이터 기반 최근 평균 점수(소수점 첫째 자리까지), type은 'actual'
       - 두 번째 아이템: name은 '다음 시험 예측', score는 향후 예상 점수(소수점 첫째 자리까지), type은 'predict'
    2. mentorComment에는 점수 산출 근거와 다음 시험을 위해 보완해야 할 점을 친절하게 요약 할 것
    """
    
    structured_llm = llm.with_structured_output(LlmPredictionResponse_Pydantic)
    response = structured_llm.invoke(prompt)
    return response
