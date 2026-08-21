"""
pipeline.py — Main MediRAG Pipeline

Integrates all modules:
  loader → chunker → embedder → vectorstore → retriever → generator

Processes multiple documents as a fault-tolerant batch. Documents are handled
sequentially because the shared embedding model and ChromaDB collection are not
assumed to be safe for concurrent writes. Replacement uploads are prepared
before the existing knowledge base is cleared, so a completely invalid upload
does not destroy a working index.

Software Engineering Principles:
  - Facade Pattern         : hides system complexity behind a simple interface
  - Dependency Injection   : modules are provided externally for flexibility and testability
  - Async/Await            : efficient handling of I/O-bound operations
  - Logging                : every major step is logged for monitoring and debugging
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from config import DATA_RAW_DIR
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.generator import Generator
from src.loader import load_document
from src.retriever import Retriever
from src.vectorstore import VectorStore

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

        self.embedder = embedder
        self.vectorstore = vectorstore or VectorStore()
        self.retriever = (
            Retriever(self.embedder, self.vectorstore) if self.embedder else None
        )
        self.generator = generator or Generator()
        self._component_lock = Lock()

        logger.info("Pipeline is ready.\n")

    # ── INGEST ───────────────────────────────────────────
    async def ingest_document(self, file_path: str) -> Dict:
        """
        Processes a single document asynchronously.
        """
        start = time.time()
        path = Path(file_path)

        embedded_chunks = await self._prepare_document(file_path)

        await asyncio.to_thread(self.vectorstore.add_chunks, embedded_chunks)

        elapsed = time.time() - start
        logger.info("%s: %s chunks — %.1fs", path.name, len(embedded_chunks), elapsed)

        return {
            "success": True,
            "file": path.name,
            "chunks": len(embedded_chunks),
            "time": elapsed,
        }

    async def _prepare_document(self, file_path: str) -> List[Dict]:
        """Load, chunk, and embed a document without changing the index."""
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Preparing: %s", path.name)

        # 1) Load
        pages = await asyncio.to_thread(load_document, file_path)
        document_id = hashlib.sha256(
            "\n".join(page["text"] for page in pages).encode("utf-8")
        ).hexdigest()[:16]
        for page in pages:
            page["metadata"]["document_id"] = document_id

        # 2) Chunk
        chunks = await asyncio.to_thread(chunk_documents, pages)

        # 3) Embed
        embedder = self._get_embedder()
        embedded_chunks = await asyncio.to_thread(embedder.embed_chunks, chunks)

        if not embedded_chunks:
            raise ValueError(f"'{path.name}' did not produce any searchable text.")

        return embedded_chunks

    def _get_embedder(self) -> Any:
        """Initialise the embedding model on its first retrieval or ingestion."""
        if self.embedder is None:
            with self._component_lock:
                if self.embedder is None:
                    self.embedder = Embedder()
        return self.embedder

    def _get_retriever(self) -> Retriever:
        """Build the retriever after the lazy embedder is available."""
        embedder = self._get_embedder()
        if self.retriever is None:
            with self._component_lock:
                if self.retriever is None:
                    self.retriever = Retriever(
                        embedder,
                        self.vectorstore,
                    )
        return self.retriever

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

    async def replace_documents(self, file_paths: List[str]) -> List[Dict]:
        """
        Replace the knowledge base with successfully prepared documents.

        All files are loaded and embedded before the current index is cleared.
        If every file fails, the existing knowledge base remains untouched.
        """
        logger.info(
            "Preparing a replacement batch of %s document(s)...", len(file_paths)
        )
        prepared_batches: List[List[Dict]] = []
        results: List[Dict] = []

        for file_path in file_paths:
            started = time.time()
            try:
                chunks = await self._prepare_document(file_path)
                prepared_batches.append(chunks)
                results.append(
                    {
                        "success": True,
                        "file": Path(file_path).name,
                        "chunks": len(chunks),
                        "time": time.time() - started,
                    }
                )
            except Exception as exc:
                logger.exception("Failed to prepare %s", file_path)
                results.append(
                    {
                        "success": False,
                        "file": Path(file_path).name,
                        "chunks": 0,
                        "error": str(exc),
                    }
                )

        if not prepared_batches:
            logger.warning("Replacement cancelled: no document could be prepared.")
            return results

        embedded_chunks = [chunk for batch in prepared_batches for chunk in batch]
        await asyncio.to_thread(self.vectorstore.clear)
        await asyncio.to_thread(self.vectorstore.add_chunks, embedded_chunks)
        logger.info(
            "Knowledge base replaced with %s document(s) and %s chunk(s).",
            len(prepared_batches),
            len(embedded_chunks),
        )
        return results

    async def ingest_directory(self, dir_path: str = DATA_RAW_DIR) -> List[Dict]:
        """
        Processes all documents in the folder sequentially.
        """
        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        supported = {".pdf", ".docx", ".txt", ".md"}
        files = [str(f) for f in path.iterdir() if f.suffix.lower() in supported]

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
                retrieval_query = rewritten_query
                logger.info("Retrieval rewrite: %s", rewritten_query)

        # 2) Retrieve
        retriever = self._get_retriever()
        retrieval_result = await asyncio.to_thread(retriever.retrieve, retrieval_query)

        # 3) Format context
        context = retriever.format_context(retrieval_result["results"])

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
        embedding_dimension = None
        if self.embedder is not None:
            if hasattr(self.embedder.model, "get_embedding_dimension"):
                embedding_dimension = self.embedder.model.get_embedding_dimension()
            else:
                embedding_dimension = (
                    self.embedder.model.get_sentence_embedding_dimension()
                )

        return {
            "total_chunks": self.vectorstore.count(),
            "embedding_dimension": embedding_dimension,
            "collection": self.vectorstore.collection.name,
        }
