"""Runtime configuration, loaded from environment variables (and an optional .env)."""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so we don't add a dependency just for this."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv()


class Settings:
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemma4:e2b-it-q4_K_M")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    DB_PATH: str = os.getenv("DB_PATH", "./data/llmmock.db")
    DEFAULT_NUM_CTX: int = int(os.getenv("DEFAULT_NUM_CTX", "8192"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "5"))
    MEMORY_MIN_SCORE: float = float(os.getenv("MEMORY_MIN_SCORE", "0.35"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))
    API_KEY: str = os.getenv("API_KEY", "")


settings = Settings()
