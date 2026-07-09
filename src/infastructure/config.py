"""Runtime configuration for the Kapruka cognitive memory stack."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load key=value pairs from the repository `.env` into `os.environ`."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "kapruka_catalog")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_CHAT_TEMPERATURE = float(os.getenv("OPENAI_CHAT_TEMPERATURE", "0.2"))
OPENAI_CHAT_MAX_TOKENS = _int_env("OPENAI_CHAT_MAX_TOKENS", 500)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


def _default_embedding_dim() -> int:
    if EMBEDDING_MODEL == "text-embedding-3-large":
        return 3072
    if EMBEDDING_MODEL == "text-embedding-3-small":
        return 1536
    return 1536


EMBEDDING_DIM = _int_env("EMBEDDING_DIM", _default_embedding_dim())
ST_MAX_TURNS = _int_env("ST_MAX_TURNS", 20)
ST_TTL_SECONDS = _int_env("ST_TTL_SECONDS", 24 * 60 * 60)
