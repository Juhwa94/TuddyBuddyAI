"""Build or replace persisted Chroma collections from UTF-8 JSONL knowledge files.

Run from the fastapi directory with: python -m rag.ingest
"""
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from common import ROOT, get_embeddings

KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
COLLECTION_FILES = {
    "prediction_policy": KNOWLEDGE_DIR / "prediction_policy.jsonl",
    "study_guidance": KNOWLEDGE_DIR / "study_guidance.jsonl",
}


def _persist_dir() -> Path:
    configured = Path(os.getenv("RAG_PERSIST_DIR", str(ROOT / "data" / "rag_index"))).expanduser()
    return configured if configured.is_absolute() else ROOT / configured


def _load_documents(path: Path, collection_name: str) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    if not path.exists():
        return documents, ids

    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                item: dict[str, Any] = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} JSON 형식 오류") from exc

            doc_id = item.get("id")
            content = item.get("content")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(f"{path.name}:{line_number} id가 필요합니다.")
            if doc_id in seen_ids:
                raise ValueError(f"{path.name}:{line_number} 중복 id: {doc_id}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{path.name}:{line_number} content가 필요합니다.")

            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"id", "content"} and isinstance(value, (str, int, float, bool))
            }
            metadata["id"] = doc_id
            metadata["knowledge_type"] = collection_name
            documents.append(Document(page_content=content.strip(), metadata=metadata))
            ids.append(doc_id)
            seen_ids.add(doc_id)
    return documents, ids


def build_indexes() -> None:
    embeddings = get_embeddings()
    persist_dir = _persist_dir()
    persist_dir.mkdir(parents=True, exist_ok=True)

    for collection_name, path in COLLECTION_FILES.items():
        documents, ids = _load_documents(path, collection_name)
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        existing_ids = store.get().get("ids", [])
        if existing_ids:
            store.delete(ids=existing_ids)
        if not documents:
            print(f"{collection_name}: 자료 없음, 기존 색인 비움")
            continue
        store.add_documents(documents=documents, ids=ids)
        print(f"{collection_name}: {len(documents)}개 자료 색인 완료")


if __name__ == "__main__":
    build_indexes()
