"""
vectorstore.py — Vector Database Module

Stores chunk embeddings in ChromaDB.
Finds the most relevant chunks via cosine-similarity search.

Chunks use deterministic IDs. Re-indexing a document replaces that source's
existing chunks, which prevents stale passages after a file is edited.
"""

import logging
from typing import Dict, List

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages persistent vector storage with ChromaDB."""

    def __init__(self):
        import chromadb
        from chromadb.config import Settings

        logger.info("Starting ChromaDB…")

        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB ready — collection '{COLLECTION_NAME}', "
            f"{self.collection.count()} chunk(s) stored.\n"
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Adds embedded chunks to ChromaDB.

        Existing chunks for the same source name are removed first, then the
        new batch is upserted. This makes document re-ingestion idempotent and
        prevents old trailing chunks from surviving when a file gets shorter.
        """
        if not chunks:
            logger.info("No chunks to add.")
            return

        # Build deterministic IDs first.
        candidate: Dict[str, Dict] = {}
        for chunk in chunks:
            m = chunk["metadata"]
            document_id = m.get("document_id", m["source"])
            cid = f"{document_id}_page_{m['page']}_chunk_{m['chunk_index']}"
            candidate[cid] = chunk

        sources = {chunk["metadata"]["source"] for chunk in candidate.values()}
        for source in sources:
            self.collection.delete(where={"source": source})

        ids, embeddings, metadatas, documents = [], [], [], []
        for cid, chunk in candidate.items():
            ids.append(cid)
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])
            documents.append(chunk["text"])

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        logger.info(
            "%s chunk(s) indexed. Total: %s",
            len(ids),
            self.collection.count(),
        )

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict]:
        """
        Returns the *top_k* chunks most similar to *query_embedding*.

        ChromaDB returns cosine distance, where lower is better. For cosine
        space, distance is approximately 1 - cosine similarity, so we expose
        a clamped similarity score in [0, 1].
        """
        total = self.collection.count()
        if total == 0 or top_k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": round(max(0.0, min(1.0, 1 - dist)), 4),
            }
            for doc, meta, dist in zip(docs, metas, distances, strict=False)
        ]

    # ── utility ───────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Deletes and recreates the collection (wipes all chunks)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared.\n")

    def count(self) -> int:
        return self.collection.count()
