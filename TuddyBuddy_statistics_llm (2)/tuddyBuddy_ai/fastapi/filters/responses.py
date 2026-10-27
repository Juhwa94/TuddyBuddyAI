"""Validate category-complete structured responses from the LLM."""
from typing import Any

from schemas import LLMFeedbackResponse_Statistics, LlmPredictionResponse_Pydantic


def _as_payload(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _signature(item: Any) -> tuple[Any, ...]:
    ids = tuple(sorted(item.categoryIds))
    return ("ids", ids) if ids else ("name", item.categoryName.strip().casefold())


def _expected_signature(group: dict[str, Any]) -> tuple[Any, ...]:
    ids = tuple(sorted(group["categoryIds"]))
    return ("ids", ids) if ids else ("name", group["categoryName"].strip().casefold())


def _validate_category_set(actual_items: list[Any], expected_groups: list[dict[str, Any]]) -> None:
    expected = [_expected_signature(group) for group in expected_groups]
    actual = [_signature(item) for item in actual_items]
    if actual != expected:
        raise ValueError("응답의 카테고리 수·식별자·순서가 입력 그룹과 일치하지 않습니다.")

    for item, group in zip(actual_items, expected_groups):
        if item.categoryName.strip().casefold() != group["categoryName"].strip().casefold():
            raise ValueError("응답의 카테고리명이 입력과 일치하지 않습니다.")


def validate_statistics_response(value: Any, expected_groups: list[dict[str, Any]]) -> LLMFeedbackResponse_Statistics:
    response = LLMFeedbackResponse_Statistics.model_validate(_as_payload(value))
    _validate_category_set(response.categoryAnalyses, expected_groups)
    for item in response.categoryAnalyses:
        if len(item.consultingPath) != 3 or not item.mentorComment.strip():
            raise ValueError("카테고리별 평가 응답 형식이 올바르지 않습니다.")
    return response


def validate_prediction_response(value: Any, expected_groups: list[dict[str, Any]]) -> LlmPredictionResponse_Pydantic:
    response = LlmPredictionResponse_Pydantic.model_validate(_as_payload(value))
    _validate_category_set(response.categoryPredictions, expected_groups)
    expected_prediction = [("최근 평균", "actual"), ("다음 시험 예측", "predict")]
    for item in response.categoryPredictions:
        actual_prediction = [(score.name, score.type) for score in item.prediction]
        if actual_prediction != expected_prediction or not item.mentorComment.strip():
            raise ValueError("카테고리별 성적예측 응답 형식이 올바르지 않습니다.")
    return response
