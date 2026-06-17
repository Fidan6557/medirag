"""
pipeline.py — Main MediRAG Pipeline

Integrates all modules:
  loader → chunker → embedder → vectorstore → retriever → generator

Processes multiple documents concurrently using asyncio:
  - Each document is embedded in a separate task
  - asyncio.gather() executes all tasks simultaneously
  - Provides significant speed improvements for large document collections

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
from typing import List, Dict

from src.loader import load_document, load_directory
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
logger = logging.getLogger(__name__)


class MediRAGPipeline:
    """
    Main pipeline class for the MediRAG system.

    Usage:
        pipeline = MediRAGPipeline()
        await pipeline.ingest_document("data/raw/medical_guide.pdf")
        result = await pipeline.ask("What is the dosage of paracetamol?")
    """

    def __init__(self):                          # ← Self deyil, self olmalıdır
        logger.info("MediRAG Pipeline is launching...")

        self.embedder    = Embedder()
        self.vectorstore = VectorStore()
        self.retriever   = Retriever(self.embedder, self.vectorstore)
        self.generator   = Generator()

        logger.info("Pipeline is ready.\n")

    # ── INGEST ───────────────────────────────────────────
    async def ingest_document(self, file_path: str) -> Dict:
        """
        Processes a single document asynchronously.
        """
        start = time.time()                      # ← start əvvəldə olmalıdır
        path = Path(file_path)                   # ← path təyin edilməmişdi

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return {"success": False, "file": file_path, "chunks": 0}

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
        Multiple documents processed in parallel using asyncio.gather().
        """
        logger.info(f"{len(file_paths)} documents are being processed in parallel...")
        start = time.time()

        tasks = [self.ingest_document(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        success = sum(1 for r in results if r["success"])
        logger.info(f"{success}/{len(file_paths)} completed — {elapsed:.1f}s\n")

        return list(results)

    async def ingest_directory(self, dir_path: str = DATA_RAW_DIR) -> List[Dict]:
        """
        Processes all documents in the folder in parallel.
        """
        supported = {".pdf", ".docx", ".txt", ".md"}
        files = [
            str(f) for f in Path(dir_path).iterdir()
            if f.suffix.lower() in supported
        ]

        if not files:
            logger.warning(f"No documents found in: {dir_path}")
            return []

        return await self.ingest_many(files)

    async def ask(self, query: str) -> Dict:
        """
        Asks a question and retrieves an answer.
        """
        logger.info(f"Question: {query}")
        start = time.time()

        # 1) Retrieve
        retrieval_result = await asyncio.to_thread(
            self.retriever.retrieve, query
        )

        # 2) Format context
        context = self.retriever.format_context(retrieval_result["results"])

        # 3) Generate
        response = await asyncio.to_thread(
            self.generator.generate, query, context, retrieval_result
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
        return {
            "total_chunks": self.vectorstore.count(),
            "model"       : self.embedder.model.get_sentence_embedding_dimension(),
            "collection"  : self.vectorstore.collection.name
        }