# 허승표

# 챗봇 테스트 결과를 Pandas로 정리하는 별도 실행 파일


import pathlib
import re
import time
from datetime import datetime

import pandas as pd
from rag.chatbot import rag_answer

ROOT = pathlib.Path(__file__).resolve().parent


# 등록된 처리 오류 표현
ERROR_PHRASES = [
    "현재 답변을 가져오지 못했습니다",
    "현재 질문을 처리하지 못했습니다",
    "일시적인 처리 오류",
    "서버 응답을 받지 못했습니다"
]


# 안내 문서에서 답을 확인할 수 없다는 표현
NO_INFO_PHRASES = [
    "안내 정보 확인 불가",
    "해당 정보를 확인할 수 없습니다",
    "해당 정보를 찾을 수 없습니다",
    "안내 정보를 찾지 못했습니다",
    "관련 정보를 찾을 수 없습니다",
    "문서에 없습니다",
    "문서에서 확인되지 않습니다"
]


# 재질문이나 질문 구체화를 요청하는 표현
CLARIFY_PHRASES = [
    "함께 입력",
    "포함해",
    "포함하여",
    "알려 주세요",
    "말씀해 주세요",
    "다시 질문",
    "구체적으로 질문",
    "어떤 기능",
    "어떤 항목",
    "어떤 대상",
    "무엇을 생성",
    "무엇을 삭제"
]


# 재질문 안내에 실제 절차가 섞였는지 확인하는 보조 표현
PROCEDURE_PHRASES = [
    "클릭합니다",
    "클릭하세요",
    "눌러 주세요",
    "선택합니다",
    "선택하세요",
    "입력합니다",
    "생성됩니다",
    "삭제됩니다"
]


TEST_CASES = [
    {
        "question": "생성 방법은?",
        "expected": "생성 대상을 포함해 다시 질문하도록 안내",
        "check_type": "CLARIFY",
        "keywords": ["생성"]
    },
    {
        "question": "삭제는 어떻게 해?",
        "expected": "삭제 대상을 포함해 다시 질문하도록 안내",
        "check_type": "CLARIFY",
        "keywords": ["삭제"]
    },
    {
        "question": "스터디룸",
        "expected": "허용된 PDF에 근거한 기본 설명",
        "check_type": "CONTENT",
        "keywords": ["스터디룸"]
    },
    {
        "question": "회원가입 방법을 알려주세요",
        "expected": "허용된 PDF에 근거한 회원가입 안내",
        "check_type": "CONTENT",
        "keywords": ["회원가입", "가입"]
    },
    {
        "question": "비밀번호를 잊었어요",
        "expected": (
            "허용된 PDF에 근거한 안내 또는 "
            "근거가 없을 경우 정보 확인 불가"
        ),
        "check_type": "CONTENT",
        "keywords": ["비밀번호"]
    },
    {
        "question": "스터디룸 생성 방법을 알려주세요",
        "expected": (
            "회원 유형에 허용된 PDF만 근거로 안내"
        ),
        "check_type": "CONTENT",
        "keywords": ["스터디룸"]
    },
    {
        "question": "오늘 서울 날씨를 알려주세요",
        "expected": "안내 문서에서 확인할 수 없다고 안내",
        "check_type": "NO_INFO",
        "keywords": []
    }
]


def normalize_text(text):
    """Markdown과 공백 차이를 줄여 비교합니다."""
    text = text.lower()

    for mark in ["*", "#", "`", "_"]:
        text = text.replace(mark, "")

    return "".join(text.split())


def contains_any(answer, phrases):
    """등록한 표현 중 하나 이상이 포함됐는지 확인합니다."""
    answer = normalize_text(answer)

    return any(
        normalize_text(phrase) in answer
        for phrase in phrases
    )


def has_numbered_steps(answer):
    """줄 시작에 '1. ' 또는 '1) ' 형태의 목록이 있는지 확인합니다."""
    return bool(
        re.search(
            r"^\s*\d+[.)]\s+",
            answer,
            flags=re.MULTILINE
        )
    )


def check_answer(test, answer, status):
    """검사 결과와 판정 이유를 반환합니다."""

    if status != "SUCCESS":
        return "실패", "챗봇 호출 또는 답변 형식 오류"

    if not answer.strip():
        return "실패", "빈 답변"

    if contains_any(answer, ERROR_PHRASES):
        return "실패", "등록된 처리 오류 안내가 포함됨"

    has_no_info = contains_any(
        answer,
        NO_INFO_PHRASES
    )

    has_clarify = contains_any(
        answer,
        CLARIFY_PHRASES
    )

    has_procedure = (
        has_numbered_steps(answer)
        or contains_any(answer, PROCEDURE_PHRASES)
    )

    check_type = test["check_type"]

    # 1. 모호한 질문에 대한 재질문 안내 검사
    if check_type == "CLARIFY":
        if has_no_info and not has_clarify:
            return (
                "실패",
                "재질문 요청 대신 정보 확인 불가 안내"
            )

        if has_no_info and has_clarify:
            return (
                "확인 필요",
                "재질문 요청과 정보 확인 불가 안내가 섞여 있음"
            )

        if not has_clarify:
            return (
                "확인 필요",
                "등록된 재질문 표현을 찾지 못함"
            )

        if not contains_any(answer, test["keywords"]):
            return (
                "확인 필요",
                "재질문 표현은 있으나 질문의 행동 표현을 찾지 못함"
            )

        if has_procedure:
            return (
                "확인 필요",
                "재질문 안내에 목록 또는 절차 표현이 포함됨"
            )

        return (
            "통과",
            "등록된 재질문 조건 충족; 임의 설명 여부는 완전 검증하지 않음"
        )

    # 2. 문서 범위 밖 질문에 대한 안내 검사
    if check_type == "NO_INFO":
        if not has_no_info:
            return (
                "확인 필요",
                "등록된 정보 확인 불가 표현을 찾지 못함"
            )

        if has_procedure:
            return (
                "확인 필요",
                "정보 확인 불가 안내와 절차 표현이 함께 있음"
            )

        # 날씨 정보를 실제로 제시했을 가능성 확인
        weather_details = [
            "맑음",
            "맑습니다",
            "흐림",
            "흐립니다",
            "비가 옵니다",
            "눈이 옵니다"
        ]

        has_temperature = bool(
            re.search(
                r"-?\d+(?:\.\d+)?\s*(?:℃|°C|도)",
                answer,
                flags=re.IGNORECASE
            )
        )

        if (
            contains_any(answer, weather_details)
            or has_temperature
        ):
            return (
                "확인 필요",
                "정보 확인 불가 안내와 실제 날씨 정보가 섞였을 가능성"
            )

        return (
            "통과",
            "등록된 정보 확인 불가 안내 조건 충족"
        )

    # 3. 기능 안내는 실제 PDF 대조 없이 정확성을 확정하지 않음
    if check_type == "CONTENT":
        if has_no_info:
            return (
                "확인 필요",
                "정보 확인 불가 안내가 적절한지 허용된 PDF와 대조 필요"
            )

        if not contains_any(answer, test["keywords"]):
            return (
                "확인 필요",
                "질문 관련 표현을 찾지 못함; 다른 표현인지 확인 필요"
            )

        return (
            "확인 필요",
            "관련 표현 확인; 내용 정확성과 회원별 공개 범위는 PDF 대조 필요"
        )

    return "실패", "테스트 설정에 없는 검사 유형"


def save_results(pd, results, output_file):
    """결과를 표로 만들고 CSV로 저장합니다."""
    df = pd.DataFrame(results)

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    return df


def main():
    try:
        import pandas as pd
    except ImportError:
        print("Pandas가 설치되어 있지 않습니다.")
        print("python -m pip install pandas")
        return

    try:
        from rag.chatbot import rag_answer
    except Exception as error:
        print("챗봇을 불러오지 못했습니다.")
        print(f"{type(error).__name__}: {error}")
        print(
            "기존 가상환경, 패키지, API 키 설정을 확인하세요."
        )
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_file = (
        ROOT / f"chatbot_test_{timestamp}.csv"
    )

    results = []
    total = len(TEST_CASES) * 2
    stopped = False

    print(f"총 {total}개 테스트를 시작합니다.")
    print("실제 AI 호출 시간과 사용량이 발생합니다.")
    print("실행 중 CSV를 열지 마세요.")
    print(
        "자동검사는 표현 검사이며, 답변 전체의 정답 판정은 아닙니다."
    )

    try:
        for user_type in ["GUEST", "MEMBER"]:
            for test in TEST_CASES:
                print(
                    f"\n[{len(results) + 1}/{total}] "
                    f"{user_type} / {test['question']}"
                )

                answer = ""
                status = "ERROR"
                error_message = ""

                start = time.perf_counter()

                try:
                    result = rag_answer(
                        question=test["question"],
                        user_type=user_type
                    )

                    if not isinstance(result, dict):
                        raise ValueError(
                            "반환값이 딕셔너리가 아닙니다."
                        )

                    received_answer = result.get("answer")

                    if (
                        not isinstance(received_answer, str)
                        or not received_answer.strip()
                    ):
                        raise ValueError(
                            "답변이 없거나 문자열이 아닙니다."
                        )

                    answer = received_answer.strip()
                    status = "SUCCESS"

                except Exception as error:
                    error_message = (
                        f"{type(error).__name__}: {error}"
                    )

                elapsed = round(
                    time.perf_counter() - start,
                    3
                )

                check_result, reason = check_answer(
                    test,
                    answer,
                    status
                )

                results.append({
                    "회원구분": user_type,
                    "질문": test["question"],
                    "기대결과": test["expected"],
                    "실제답변": answer,
                    "응답시간_초": elapsed,
                    "실행상태": status,
                    "오류내용": error_message,
                    "자동검사결과": check_result,
                    "판정이유": reason,
                    "검토메모": ""
                })

                print(f"실행 상태: {status}")
                print(f"응답 시간: {elapsed}초")
                print(f"자동 검사: {check_result}")
                print(f"판정 이유: {reason}")

                if status == "SUCCESS":
                    print(f"답변:\n{answer}")
                else:
                    print(f"오류: {error_message}")

                # 완료된 항목마다 저장
                try:
                    save_results(
                        pd,
                        results,
                        output_file
                    )

                except OSError as error:
                    print(f"\nCSV 저장 실패: {error}")
                    print(
                        "추가 호출을 중단하고 종료 시 저장을 재시도합니다."
                    )
                    stopped = True
                    break

            if stopped:
                break

    except KeyboardInterrupt:
        print("\n테스트를 중단했습니다.")
        print("완료된 결과만 저장합니다.")

    if not results:
        print("저장할 결과가 없습니다.")
        return

    df = pd.DataFrame(results)

    try:
        save_results(
            pd,
            results,
            output_file
        )
        print(f"\n결과 저장 완료:\n{output_file}")

    except OSError as error:
        print(f"\n최종 CSV 저장 실패: {error}")
        print("아래 결과를 복사해 보관하세요.")
        print(df.to_string(index=False))

    print("\n===== 결과 미리보기 =====")
    print(
        df[
            [
                "회원구분",
                "질문",
                "자동검사결과",
                "판정이유"
            ]
        ].head()
    )

    print("\n===== 데이터 정보 =====")
    df.info()

    print("\n===== 결측값 확인 =====")
    print(df.isnull().sum())

    print("\n===== 자동검사 집계 =====")

    for label in ["통과", "실패", "확인 필요"]:
        count = len(
            df[df["자동검사결과"] == label]
        )
        print(f"{label}: {count}개")

    print(
        "\n빈 문자열은 isnull()에서 결측값으로 세지 않습니다."
    )
    print(
        "통과는 등록한 조건 충족을 의미합니다. "
        "회원별 정보 공개 범위와 PDF 내용은 직접 확인하세요."
    )


if __name__ == "__main__":
    main()