"""
config.py — MediRAG Configuration
"""

import os
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

T = TypeVar("T", int, float)


def _project_path(value: str) -> str:
    """Resolve relative configuration paths from the project directory."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else BASE_DIR / path)


def _number_from_env(
    name: str,
    default: T,
    cast: Callable[[str], T],
    *,
    minimum: T,
    maximum: T | None = None,
) -> T:
    """Read and validate a numeric environment setting."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = cast(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid number, got {raw_value!r}.") from exc

    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"at least {minimum}"
        if maximum is not None:
            bounds = f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {bounds}, got {value}.")
    return value


def _bool_from_env(name: str, default: bool = False) -> bool:
    """Read a conventional true/false environment setting."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalised = raw_value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw_value!r}.")


# ── LLM ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ── Embedding ───────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)


# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE = _number_from_env("CHUNK_SIZE", 512, int, minimum=64)
CHUNK_OVERLAP = _number_from_env("CHUNK_OVERLAP", 50, int, minimum=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


# ── Vector Store ────────────────────────────────────────
CHROMA_PERSIST_DIR = _project_path(os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medirag_docs")


# ── Retrieval ───────────────────────────────────────────
TOP_K = _number_from_env("TOP_K", 3, int, minimum=1, maximum=20)
CONFIDENCE_THRESHOLD = _number_from_env(
    "CONFIDENCE_THRESHOLD",
    0.30,
    float,
    minimum=0.0,
    maximum=1.0,
)


# ── Supported Languages ─────────────────────────────────
SUPPORTED_LANGUAGES = ["en", "az", "ru"]


# ── Data Paths ──────────────────────────────────────────
DATA_RAW_DIR = str(BASE_DIR / "data" / "raw")


# ── Application ─────────────────────────────────────────
MAX_FILE_SIZE_MB = _number_from_env(
    "MAX_FILE_SIZE_MB",
    25,
    int,
    minimum=1,
    maximum=500,
)
SERVER_NAME = os.getenv("SERVER_NAME", "127.0.0.1")
SERVER_PORT = _number_from_env(
    "SERVER_PORT",
    7860,
    int,
    minimum=1,
    maximum=65535,
)
GRADIO_SHARE = _bool_from_env("GRADIO_SHARE", False)
