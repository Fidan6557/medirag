"""
config.py — MediRAG Configuration
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _project_path(value: str) -> str:
    """Resolve relative configuration paths from the project directory."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else BASE_DIR / path)


# ── LLM ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ── Embedding ───────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50


# ── Vector Store ────────────────────────────────────────
CHROMA_PERSIST_DIR = _project_path(
    os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medirag_docs")


# ── Retrieval ───────────────────────────────────────────
TOP_K                = 3
CONFIDENCE_THRESHOLD = 0.30


# ── Supported Languages ─────────────────────────────────
SUPPORTED_LANGUAGES = ["en", "az", "ru"]


# ── Data Paths ──────────────────────────────────────────
DATA_RAW_DIR = str(BASE_DIR / "data" / "raw")
