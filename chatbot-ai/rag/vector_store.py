# 허승표
# 나눠진 문서 검색 담당
# Chunks => Embedding => Vector DB 저장

import sys
import pathlib

sys.path.append(
    str(pathlib.Path(__file__).resolve().parent.parent)
)

from rag.common import get_embeddings, DATA
from rag.document_loader import load_and_split_documents
from langchain_community.vectorstores import FAISS


# 저장할 인덱스의 위치
INDEX_DIR = DATA / "policy_faiss_index"


def build_vector_store():
    
    # PDF를 읽고 청킹
    chunks = load_and_split_documents()

    # Gemini Embedding 모델 가져오기
    embeddings = get_embeddings()
 
    # FAISS 벡터 DB 생성
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )
    
    # FAISS 인덱스를 파일로 저장
    vector_store.save_local(str(INDEX_DIR))

    # 테스트용
    print(f"생성된 청크 수 : {len(chunks)}")
    print(f"FAISS 저장 완료 : {INDEX_DIR}")

    return vector_store


def search_documents(query: str, k: int = 3, user_type: str = "GUEST"):
    
    # Embedding 모델 준비
    embeddings = get_embeddings()
    
    # 저장된 FAISS 불러오기
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 회원/비회원 검색 범위 설정
    if user_type == "MEMBER":
        access_filter = {
            "access_level" : {"$in" : ["COMMON", "MEMBER"]}
        } 
    else : 
        access_filter = {
            "access_level" : "COMMON"    
        }
    
    # 회원 유형에 맞는 청크 검색
    results = vector_store.similarity_search(
        query,
        k = k,
        filter = access_filter,
        fetch_k = len(vector_store.index_to_docstore_id)
    )
    
    return results


if __name__ == "__main__":

    # 실제 정책 PDF로 FAISS 인덱스 생성
    build_vector_store()

    query = "스터디룸은 어떻게 생성하나요?"

    # 비회원 검색 테스트
    print("\n===== 비회원 검색 =====")

    guest_results = search_documents(
        query,
        user_type="GUEST"
    )

    for i, doc in enumerate(guest_results):
        print(f"\n검색결과 {i + 1}")
        print(f"파일: {pathlib.Path(doc.metadata['source']).name}")
        print(f"접근 범위: {doc.metadata['access_level']}")
        print(doc.page_content)
    
    # 회원 검색 테스트
    print("\n===== 회원 검색 =====")

    member_results = search_documents(
        query,
        user_type="MEMBER"
    )

    for i, doc in enumerate(member_results):
        print(f"\n검색결과 {i + 1}")
        print(f"파일: {pathlib.Path(doc.metadata['source']).name}")
        print(f"접근 범위: {doc.metadata['access_level']}")
        print(doc.page_content)

