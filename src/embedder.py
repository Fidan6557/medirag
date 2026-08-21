"""
Encodes each text chunk into a dense embedding vector using a configurable
sentence-transformer model.

Model characteristics:
  - Embedding size: 384 dimensions
  - Fast inference speed
  - Low memory footprint
  - Multilingual support
"""

import logging
from typing import Dict, List

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedder class responsible for generating vector embeddings
    from text chunks.

    The model is initialized once and reused throughout the
    application, reducing loading time and improving overall
    performance.
    """

    def __init__(self):
        # Importing sentence-transformers also imports the ML runtime. Keeping
        # it here makes the web interface start quickly and loads the model only
        # when the first document is processed.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s...", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")

    def embed_text(self, text: str) -> List[float]:
        """
        Converts a single text into a vector embedding.

        Used during retrieval to embed the user's query before
        performing similarity search.
        """
        vector = self.model.encode(text)
        return vector.tolist()

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Converts all chunks into vector embeddings.

        An 'embedding' field is added to each chunk.

        Input:  Output from chunker.py — List[Dict]
        Output: Same List[Dict] with an additional 'embedding' field
        """
        logger.info("Embedding %s chunk(s)...", len(chunks))
        if not chunks:
            logger.info("No chunks to embed.")
            return []

        texts = [chunk["text"] for chunk in chunks]

        # batch_size=32 — process 32 chunks at a time for faster embedding generation
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
        )

        # Add embeddings to each chunk
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk["embedding"] = embedding.tolist()

        logger.info(
            "%s chunk(s) embedded (%s dimensions).",
            len(chunks),
            len(embeddings[0]),
        )

        return chunks
