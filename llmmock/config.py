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
    # How long ollama keeps a model in memory after a request. Duration string ("30m",
    # "1h"), seconds ("300"), "-1" = keep forever, "0" = unload immediately, "" = disable.
    KEEP_ALIVE: str = os.getenv("KEEP_ALIVE", "30m")
    # Preload DEFAULT_MODEL into memory at startup so the first request isn't a cold start.
    PRELOAD: bool = os.getenv("PRELOAD", "true").lower() in ("1", "true", "yes", "on")


settings = Settings()


def keep_alive_value():
    """settings.KEEP_ALIVE as int seconds / -1 when numeric, else the raw duration string,
    or None to omit it (use ollama's own default)."""
    v = (settings.KEEP_ALIVE or "").strip()
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return v
