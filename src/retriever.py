"""
retriever.py — Retrieval Module

Embeds the user's query and searches for relevant chunks in the VectorStore.
Filters weak matches using a confidence threshold.

This is the foundation of out-of-scope detection:
  - If the score is high  → answer the question
  - If the score is low   → respond with "not found in the document"
"""

from typing import Dict, List, Tuple
from src.embedder import Embedder
from src.vectorstore import VectorStore
from config import TOP_K, CONFIDENCE_THRESHOLD


class Retriever:
    """
    Retriever class — receives a user query and returns the most relevant chunks.
    """

    def __init__(self, embedder: Embedder, vectorstore: VectorStore):
        # Dependency injection — the embedder and vector store are provided externally.
        # This makes testing much easier, since they can be mocked.
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = CONFIDENCE_THRESHOLD
    ) -> Dict:
        """
        Finds the most relevant chunks for a given query.

        Returns:
            {
                "query": "...",
                "results": [...],      # chunks above the threshold
                "is_answerable": True, # whether it is possible to answer the question
                "best_score": 0.87     # highest similarity score among results
            }
        """ 
        # 1) Convert the question into a vector (embedding)
        query_embedding = self.embedder.embed_text(query)

        # 2) Search in the VectorStore 
        raw_results = self.vectorstore.search(query_embedding, top_k=top_k)

        if not raw_results:
            return {
                "query": query,
                "results": [],
                "is_answerable": False,
                "best_score": 0.0
            }
        # 3) Apply threshold filter — remove weak results
        filtered = [r for r in raw_results if r["score"] >= threshold]

        best_score = max(r["score"] for r in filtered) if filtered else 0.0
        is_answerable = len(filtered) > 0

        if not is_answerable:
            print(f"Question not found in the document. (best score: {best_score:.2f} < {threshold})")
        else:
            print(f"{len(filtered)} relevant chunks found. (best score: {best_score:.2f})")

        return {
            "query": query,
            "results": filtered,
            "is_answerable": is_answerable,
            "best_score": best_score
        }

    def format_context(self, results: List[Dict]) -> str:
        """
        Converts chunks into a context string for the generator.
        Also attaches the source information for each chunk.
        """
        context_parts = []
   
        for i, r in enumerate(results, 1):
            source = r["metadata"]["source"]
            page = r["metadata"]["page"]
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{r['text']}"
            )

        return "\n\n".join(context_parts)
        
