from typing import List

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import AnalyzeRequest, CategoryStatItem
from service.llm_service import analyze_score_prediction, analyze_user_statistics

app = FastAPI()


@app.post("/statistics", status_code=status.HTTP_200_OK)
async def analyze_performance(request: AnalyzeRequest):
    if not request.quiz_logs:
        return {"categoryAnalyses": []}

    try:
        quiz_logs_dict = [log.model_dump() for log in request.quiz_logs]
        return analyze_user_statistics(request.user_id, quiz_logs_dict)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(exc)},
        )


@app.post("/prediction", status_code=status.HTTP_200_OK)
async def analyze_prediction_endpoint(stats_list: List[CategoryStatItem]):
    if not stats_list:
        return {"categoryPredictions": []}

    try:
        stats_dict_list = [stat.model_dump() for stat in stats_list]
        return analyze_score_prediction(stats_dict_list)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(exc)},
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )
