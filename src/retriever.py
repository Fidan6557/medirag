"""
retriever.py — Retrieval Module

Embeds the user's query and searches for relevant chunks in the VectorStore.
Filters weak matches using a confidence threshold.

This is the foundation of out-of-scope detection:
  - If the score is high  → answer the question
  - If the score is low   → respond with "not found in the document"
"""

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Set

from config import CONFIDENCE_THRESHOLD, TOP_K

if TYPE_CHECKING:
    from src.embedder import Embedder
    from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b[\w]+\b", re.UNICODE)
_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "used",
    "use",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "bu",
    "bir",
    "ile",
    "ilə",
    "ne",
    "nə",
    "nedir",
    "nədir",
    "ucun",
    "üçün",
    "istifade",
    "istifadə",
    "olunur",
    "edir",
    "в",
    "и",
    "как",
    "на",
    "о",
    "об",
    "от",
    "по",
    "что",
    "это",
}


def _tokenise(text: str) -> Set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = _tokenise(query)
    if not query_tokens:
        return 0.0

    text_tokens = _tokenise(text)
    if not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


class Retriever:
    """
    Retriever class — receives a user query and returns the most relevant chunks.
    """

    def __init__(self, embedder: "Embedder", vectorstore: "VectorStore"):
        # Dependency injection — the embedder and vector store are provided externally.
        # This makes testing much easier, since they can be mocked.
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(
        self, query: str, top_k: int = TOP_K, threshold: float = CONFIDENCE_THRESHOLD
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

        # 2) Search in the VectorStore. Pull extra candidates so lexical
        # reranking can rescue exact matches that semantic search placed lower.
        search_k = max(top_k * 4, top_k)
        raw_results = self.vectorstore.search(query_embedding, top_k=search_k)

        if not raw_results:
            return {
                "query": query,
                "results": [],
                "is_answerable": False,
                "best_score": 0.0,
            }
        # 3) Rerank using semantic score + exact keyword overlap.
        ranked_results = []
        for result in raw_results:
            lexical_score = _lexical_overlap(query, result["text"])
            combined_score = (result["score"] * 0.75) + (lexical_score * 0.25)
            length_factor = min(1.0, max(0.60, len(result["text"].split()) / 80))
            combined_score *= length_factor
            ranked_results.append(
                {
                    **result,
                    "semantic_score": result["score"],
                    "lexical_score": round(lexical_score, 4),
                    "length_factor": round(length_factor, 4),
                    "score": round(combined_score, 4),
                }
            )

        ranked_results.sort(key=lambda r: r["score"], reverse=True)

        if any(r["lexical_score"] > 0 for r in ranked_results):
            ranked_results = [r for r in ranked_results if r["lexical_score"] > 0]

        # 4) Apply threshold filter — remove weak results
        raw_best_score = ranked_results[0]["score"]
        min_relevant_score = max(threshold, raw_best_score - 0.10)
        filtered = [r for r in ranked_results if r["score"] >= min_relevant_score][
            :top_k
        ]

        best_score = raw_best_score
        is_answerable = len(filtered) > 0

        if not is_answerable:
            logger.info(
                "Question not found in the document (best score: %.2f < %.2f).",
                best_score,
                threshold,
            )
        else:
            logger.info(
                "%s relevant chunk(s) found (best score: %.2f).",
                len(filtered),
                best_score,
            )

        return {
            "query": query,
            "results": filtered,
            "is_answerable": is_answerable,
            "best_score": best_score,
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
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{r['text']}")

        return "\n\n".join(context_parts)
