from pathlib import Path
from pydantic import BaseModel, SecretStr
from typing import List
import os
import json
import logging
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).parent
_env_path = _BASE_DIR / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


class Settings(BaseModel):
    API_VERSION: str = "4.2.0"
    API_TITLE: str = "Document Research Assistant"

    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))
    MAX_TOTAL_FILES: int = int(os.getenv("MAX_TOTAL_FILES", "50"))
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt"]

    UPLOAD_DIR: Path = _BASE_DIR / "uploads"
    VECTOR_STORE_DIR: Path = _BASE_DIR / "vector_store"
    CONVERSATION_DIR: Path = _BASE_DIR / "conversations"
    LOG_DIR: Path = _BASE_DIR / "logs"

    VECTOR_DIMENSION: int = 768

    MODEL_NAME: str = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    HF_TOKEN: SecretStr = SecretStr(os.getenv("HF_TOKEN", ""))

    OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
    PAGEINDEX_MODEL: str = os.getenv("PAGEINDEX_MODEL", "gpt-4o-2024-11-20")
    PAGEINDEX_MAX_CONCURRENT: int = int(os.getenv("PAGEINDEX_MAX_CONCURRENT", "5"))
    PAGEINDEX_STORE_DIR: Path = _BASE_DIR / "page_index_store"
    PAGEINDEX_TOC_CHECK_PAGES: int = int(os.getenv("PAGEINDEX_TOC_CHECK_PAGES", "20"))
    PAGEINDEX_MAX_PAGES_PER_NODE: int = int(os.getenv("PAGEINDEX_MAX_PAGES_PER_NODE", "10"))
    PAGEINDEX_MAX_TOKENS_PER_NODE: int = int(os.getenv("PAGEINDEX_MAX_TOKENS_PER_NODE", "20000"))

    RATE_LIMIT_INDEX: str = os.getenv("RATE_LIMIT_INDEX", "5/minute")
    RATE_LIMIT_SEARCH: str = os.getenv("RATE_LIMIT_SEARCH", "30/minute")
    RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "10/minute")

    CORS_ORIGINS: List[str] = ["http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = False

    MAX_CONVERSATION_AGE: int = 24 * 60 * 60

    class Config:
        arbitrary_types_allowed = True


def _parse_list_env(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key, "")
    if not val:
        return default
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [v.strip() for v in val.split(",") if v.strip()]


settings = Settings(
    CORS_ORIGINS=_parse_list_env("CORS_ORIGINS", ["http://localhost:8000"]),
    ALLOWED_EXTENSIONS=_parse_list_env("ALLOWED_EXTENSIONS", [".pdf", ".docx", ".txt"]),
)

for _dir in [
    settings.UPLOAD_DIR,
    settings.VECTOR_STORE_DIR,
    settings.CONVERSATION_DIR,
    settings.LOG_DIR,
    settings.PAGEINDEX_STORE_DIR,
]:
    _dir.mkdir(exist_ok=True)


def validate_environment() -> List[str]:
    warnings: List[str] = []
    if not settings.HF_TOKEN.get_secret_value():
        warnings.append("WARNING: HF_TOKEN not set. Some models may not be accessible.")
    if not settings.OPENAI_API_KEY.get_secret_value():
        warnings.append("WARNING: OPENAI_API_KEY not set. PageIndex RAG will be unavailable.")
    return warnings


for _w in validate_environment():
    logging.warning(_w)