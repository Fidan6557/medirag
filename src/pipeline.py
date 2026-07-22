"""
pipeline.py — Main MediRAG Pipeline

Integrates all modules:
  loader → chunker → embedder → vectorstore → retriever → generator

Processes multiple documents as a fault-tolerant batch. Documents are handled
sequentially because the shared embedding model and ChromaDB collection are not
assumed to be safe for concurrent writes.

Software Engineering Principles:
  - Facade Pattern         : hides system complexity behind a simple interface
  - Dependency Injection   : modules are provided externally for flexibility and testability
  - Async/Await            : efficient handling of I/O-bound operations
  - Logging                : every major step is logged for monitoring and debugging
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.loader import load_document
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.generator import Generator
from config import DATA_RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
for noisy_logger in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class MediRAGPipeline:
    """
    Main pipeline class for the MediRAG system.

    Usage:
        pipeline = MediRAGPipeline()
        await pipeline.ingest_document("data/raw/medical_guide.pdf")
        result = await pipeline.ask("What is the dosage of paracetamol?")
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        vectorstore: Optional[Any] = None,
        generator: Optional[Any] = None,
    ):
        logger.info("MediRAG Pipeline is launching...")

        self.embedder    = embedder or Embedder()
        self.vectorstore = vectorstore or VectorStore()
        self.retriever   = Retriever(self.embedder, self.vectorstore)
        self.generator   = generator or Generator()

        logger.info("Pipeline is ready.\n")

    # ── INGEST ───────────────────────────────────────────
    async def ingest_document(self, file_path: str) -> Dict:
        """
        Processes a single document asynchronously.
        """
        start = time.time()
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing: {path.name}")

        # 1) Load
        pages = load_document(file_path)

        # 2) Chunk
        chunks = chunk_documents(pages)

        # 3) Embed
        embedded_chunks = await asyncio.to_thread(
            self.embedder.embed_chunks, chunks
        )

        # 4) Write to VectorStore
        await asyncio.to_thread(
            self.vectorstore.add_chunks, embedded_chunks
        )

        elapsed = time.time() - start
        logger.info(f"{path.name}: {len(embedded_chunks)} chunks — {elapsed:.1f}s")

        return {
            "success": True,
            "file"   : path.name,
            "chunks" : len(embedded_chunks),
            "time"   : elapsed
        }

    async def ingest_many(self, file_paths: List[str]) -> List[Dict]:
        """
        Processes multiple documents without aborting after one bad file.
        """
        logger.info(f"Processing a batch of {len(file_paths)} document(s)...")
        start = time.time()

        async def ingest_safely(file_path: str) -> Dict:
            try:
                return await self.ingest_document(file_path)
            except Exception as exc:
                logger.exception("Failed to ingest %s", file_path)
                return {
                    "success": False,
                    "file": Path(file_path).name,
                    "chunks": 0,
                    "error": str(exc),
                }

        results = []
        for file_path in file_paths:
            results.append(await ingest_safely(file_path))

        elapsed = time.time() - start
        success = sum(1 for r in results if r["success"])
        logger.info(f"{success}/{len(file_paths)} completed — {elapsed:.1f}s\n")

        return list(results)

    async def ingest_directory(self, dir_path: str = DATA_RAW_DIR) -> List[Dict]:
        """
        Processes all documents in the folder in parallel.
        """
        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        supported = {".pdf", ".docx", ".txt", ".md"}
        files = [
            str(f) for f in path.iterdir()
            if f.suffix.lower() in supported
        ]

        if not files:
            logger.warning(f"No documents found in: {dir_path}")
            return []

        return await self.ingest_many(files)

    async def ask(
        self,
        query: str,
        response_instruction: Optional[str] = None,
    ) -> Dict:
        """
        Asks a question and retrieves an answer.
        """
        logger.info(f"Question: {query}")
        start = time.time()

        query = query.strip()
        if not query:
            raise ValueError("Question cannot be empty.")

        if self.vectorstore.count() == 0:
            return {
                "answer": "No documents have been indexed yet.",
                "sources": [],
                "score": 0.0,
                "answered": False,
                "time": time.time() - start,
            }

        # 1) Build retrieval query
        retrieval_query = query
        if self.generator.should_rewrite_query_for_retrieval(query):
            rewritten_query = await asyncio.to_thread(
                self.generator.rewrite_query_for_retrieval,
                query,
            )
            if rewritten_query != query:
                retrieval_query = f"{query}\n{rewritten_query}"
                logger.info(f"Retrieval rewrite: {rewritten_query}")

        # 2) Retrieve
        retrieval_result = await asyncio.to_thread(
            self.retriever.retrieve, retrieval_query
        )

        # 3) Format context
        context = self.retriever.format_context(retrieval_result["results"])

        # 4) Generate
        response = await asyncio.to_thread(
            self.generator.generate,
            query,
            context,
            retrieval_result,
            response_instruction,
        )

        elapsed = time.time() - start
        response["time"] = elapsed

        logger.info(f"Answer ready — {elapsed:.1f}s\n")

        return response

    # ── UTILITY ──────────────────────────────────────────
    def clear_knowledge_base(self) -> None:
        """Removes all documents from the vector database."""
        self.vectorstore.clear()
        logger.info("Knowledge base cleared.")

    def get_stats(self) -> Dict:
        """Pipeline statistics."""
        if hasattr(self.embedder.model, "get_embedding_dimension"):
            embedding_dimension = self.embedder.model.get_embedding_dimension()
        else:
            embedding_dimension = self.embedder.model.get_sentence_embedding_dimension()

        return {
            "total_chunks": self.vectorstore.count(),
            "model"       : embedding_dimension,
            "collection"  : self.vectorstore.collection.name
        }
