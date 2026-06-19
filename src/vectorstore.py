"""
vectorstore.py — Vector Database Module

Stores chunk embeddings in ChromaDB.
Finds the most relevant chunks using similarity search.

How ChromaDB works:
  - Stores vectors on disk (./chroma_db/)
  - Performs search using cosine similarity
  - Also stores metadata (source, page, chunk_index)
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict 
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K


class VectorStore:
    """
    VectorStore class responsible"""

    def __init__(self):
        print("ChromaDB is starting...")

        # ChromaDB client — writes data to disk
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        # Collection — similar to a table in SQL
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
        )

        print(f" ChromaDB is ready.")
        print(f" Collection: {COLLECTION_NAME}")
        print(f"   Existing chunk count: {self.collection.count()}\n")

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Adds embedded chunks to ChromaDB.

        For each chunk, the following information is stored:
            - id         : unique identifier (source + chunk_index)
            - embedding  : vector representation of the chunk
            - metadata   : source, page, format, chunk_index
            - document   : original text content
        """
        if not chunks:
            print("No chunks to add to the vector store.")
            return
        
        # Skip chunks that already exist in the database
        existing = set(self.collection.get()["ids"])

        ids, embeddings, metadatas, documents = [], [], [], []

        for chunk in chunks:
            # Create a unique ID for each chunk based on source and chunk index
            chunk_id = f"{chunk['metadata']['source']}_page_{chunk['metadata']['page']}_chunk_{chunk['metadata']['chunk_index']}"

            if chunk_id in existing:
                continue  # Skip if this chunk already exists

            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])
            documents.append(chunk["text"])

        if not ids:
            print(" All chunks already exist in the vector store. No new chunks added.")
            return

        # Add data to ChromaDB in batches
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        print(f" {len(ids)} new chunks added to the vector store.")
        print(f"  Total chunk count: {self.collection.count()}\n")


    def search(self, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict]:
        """
        Finds the chunks that are most similar to the query vector.

        Returns:
            [
                {
                    "text": "...",
                    "metadata": {...},
                    "score": 0.87  # similarity score (0–1)
                },
                ...
            ]
        """
        total_chunks = self.collection.count()
        if total_chunks == 0 or top_k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"]
        )

        # Process and format retrieved results
        chunks = []
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        for doc, meta, dist in zip(
            documents[0],
            metadatas[0],
            distances[0]
        ):
            # ChromaDB returns cosine distance (0 = identical, 2 = completely different)
            # We convert it to similarity (1 = identical, 0 = completely different)
            score = 1 - (dist / 2)

            chunks.append({
                "text": doc,
                "metadata": meta,
                "score": score
            })

        return chunks
    
    def clear(self) -> None:
        """
        Deletes all chunks from the database.
        Used when uploading a new document to reset the collection.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared. All chunks deleted.\n")

    def count(self) -> int:
        return self.collection.count()
