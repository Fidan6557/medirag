"""
vectorstore.py — Vector Database Module

Stores chunk embeddings in ChromaDB.
Finds the most relevant chunks via cosine-similarity search.

Key fix vs original:
  - add_chunks() previously called collection.get() with no filter, which
    loads every stored embedding into RAM on large databases.
  - Now it builds the new IDs first, then asks ChromaDB for only those
    specific IDs — O(batch) instead of O(total collection).
"""

import logging
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages persistent vector storage with ChromaDB."""

    def __init__(self):
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
        Skips any chunk whose ID is already present (idempotent).
        """
        if not chunks:
            logger.info("No chunks to add.")
            return

        # Build candidate IDs first
        candidate: Dict[str, Dict] = {}
        for chunk in chunks:
            m = chunk["metadata"]
            cid = (
                f"{m['source']}_page_{m['page']}"
                f"_chunk_{m['chunk_index']}"
            )
            candidate[cid] = chunk

        # Ask ChromaDB only for the IDs we care about (cheap)
        try:
            existing_result = self.collection.get(
                ids=list(candidate.keys()),
                include=[],          # we only need the IDs, not the data
            )
            existing_ids = set(existing_result["ids"])
        except Exception:
            # get() with explicit ids never returns "not found" as an error,
            # but guard anyway
            existing_ids = set()

        new_ids, embeddings, metadatas, documents = [], [], [], []

        for cid, chunk in candidate.items():
            if cid in existing_ids:
                continue
            new_ids.append(cid)
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])
            documents.append(chunk["text"])

        if not new_ids:
            logger.info("All chunks already exist — nothing new to add.")
            return

        self.collection.add(
            ids=new_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        logger.info(
            f"{len(new_ids)} new chunk(s) added. "
            f"Total: {self.collection.count()}\n"
        )

    # ── read ──────────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: List[float], top_k: int = TOP_K
    ) -> List[Dict]:
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

        docs      = (results.get("documents") or [[]])[0]
        metas     = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        return [
            {
                "text":     doc,
                "metadata": meta,
                "score":    round(max(0.0, min(1.0, 1 - dist)), 4),
            }
            for doc, meta, dist in zip(docs, metas, distances)
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
