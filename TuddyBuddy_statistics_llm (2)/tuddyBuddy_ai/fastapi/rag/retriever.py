"""RAG retrieval helpers backed by the persisted Chroma collections."""
import logging
import os
from functools import lru_cache
from pathlib import Path

from common import ROOT, get_embeddings

logger = logging.getLogger(__name__)
COLLECTIONS = {"prediction_policy", "study_guidance"}


def _persist_dir() -> Path:
    configured = Path(os.getenv("RAG_PERSIST_DIR", str(ROOT / "data" / "rag_index"))).expanduser()
    return configured if configured.is_absolute() else ROOT / configured


@lru_cache(maxsize=2)
def _get_store(collection_name: str):
    from langchain_community.vectorstores import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(_persist_dir()),
    )


def retrieve_context(collection_name: str, query: str, top_k: int = 3) -> str:
    """Return a compact set of retrieved guidance; empty when no index is available."""
    if collection_name not in COLLECTIONS:
        raise ValueError(f"알 수 없는 RAG collection: {collection_name}")
    if not query.strip() or not _persist_dir().exists():
        return ""

    try:
        documents = _get_store(collection_name).similarity_search(query, k=top_k)
    except Exception:
        # Keep the existing endpoint usable while a collection is not yet built.
        logger.warning("RAG retrieval unavailable for collection '%s'; continuing without context", collection_name)
        return ""

    blocks = []
    for document in documents:
        doc_id = document.metadata.get("id", "unknown")
        blocks.append(f"[자료 ID: {doc_id}]\n{document.page_content.strip()}")
    return "\n\n".join(blocks)
