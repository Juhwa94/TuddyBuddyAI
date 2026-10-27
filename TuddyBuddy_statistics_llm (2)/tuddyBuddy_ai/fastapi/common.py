"""Shared model configuration and provider factories."""
import os
import pathlib
from dotenv import load_dotenv

# Application root: D:/ictstudy/tuddyBuddy_ai
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Load the app-level .env without printing or copying secret values.
load_dotenv(ROOT / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")


def require_key(name: str) -> str:
    """Return a configured API key or raise a setup error."""
    val = os.getenv(name)
    if not val or val.startswith("여기에"):
        raise SystemExit(
            f"[설정 필요] {name} 가 .env 에 없습니다.\n"
            f" 1) cp .env.example .env\n"
            f" 2) .env 파일을 열어 {name} 값을 채우세요."
        )
    return val


def get_chat(provider: str = "gemini", temperature: float = 0.0):
    """Return a LangChain chat model. provider: 'gemini' (default) | 'openai'."""
    if provider == "gemini":
        require_key("GOOGLE_API_KEY")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
    if provider == "openai":
        require_key("OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    raise ValueError(f"알 수 없는 provider: {provider}")


def get_embeddings():
    """Return the configured Gemini embedding model for the RAG modules."""
    require_key("GOOGLE_API_KEY")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)
