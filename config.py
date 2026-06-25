"""
config.py — MediRAG Configuration
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


# ── LLM ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    print(
        "ERROR: GROQ_API_KEY is not set.\n"
        "  1) Copy .env.example to .env\n"
        "  2) Add your key from https://console.groq.com\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Embedding ───────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50


# ── Vector Store ────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "medirag_docs"


# ── Retrieval ───────────────────────────────────────────
TOP_K                = 3
CONFIDENCE_THRESHOLD = 0.30


# ── Supported Languages ─────────────────────────────────
SUPPORTED_LANGUAGES = ["en", "az", "ru"]


# ── Data Paths ──────────────────────────────────────────
DATA_RAW_DIR = "./data/raw"
