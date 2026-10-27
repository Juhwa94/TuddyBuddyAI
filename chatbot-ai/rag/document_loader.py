# 허승표
# 정책 PDF 준비 담당
# PDF파일 읽기 => Chunk로 분할


import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.common import DOCS


def load_and_split_documents():

    # 회원/비회원 문서 폴더 구분
    folders = {
        "COMMON" : DOCS / "common",
        "MEMBER" : DOCS / "member"
    }
    
    docs = []
    
    # 각 폴더의 PDF 파일 검색
    for access_level, folder_path in folders.items():
        if not folder_path.exists():
            raise FileNotFoundError(f"[폴더 없음] {folder_path}")
        
        pdf_files = sorted(folder_path.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            
            print(f"[PDF 로딩] {pdf_path.name} / {access_level}")

            # PDF 로딩
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()
            
            # 문서 접근 범위 설정
            for doc in pdf_docs:
                doc.metadata["access_level"] = access_level
            
            docs.extend(pdf_docs)
    if not docs:
        raise ValueError("[PDF 없음] 로딩할 정책 PDF가 없습니다.")

    # 청크 설정
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # 문서를 청킹
    chunks = splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":

    chunks = load_and_split_documents()

    print(f"\n총 청크 개수: {len(chunks)}")

    # 로딩된 PDF 파일 확인
    pdf_files = set()

    for chunk in chunks:
        filename = pathlib.Path(chunk.metadata["source"]).name
        access_level = chunk.metadata["access_level"]

        pdf_files.add((filename, access_level))

    print(f"총 PDF 파일 개수: {len(pdf_files)}")

    for filename, access_level in sorted(pdf_files):
        print(f"{filename} / {access_level}")

