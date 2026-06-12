"""
Encodes each text chunk into a dense embedding vector using
the all-MiniLM-L6-v2 sentence transformer model.

Model characteristics:
  - Embedding size: 384 dimensions
  - Fast inference speed
  - Low memory footprint
  - Multilingual support
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
from config import EMBEDDING_MODEL 


class Embedder:
    """
    Embedder class responsible for generating vector embeddings
    from text chunks.

    The model is initialized once and reused throughout the
    application, reducing loading time and improving overall
    performance.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Model loaded successfully.")

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
        print(f"Embedding {len(chunks)} chunks...")

        texts = [chunk["text"] for chunk in chunks]

        # batch_size=32 — process 32 chunks at a time for faster embedding generation
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )

        # Add embeddings to each chunk
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        print(f"{len(chunks)} chunks embedded successfully.")
        print(f"   Vector size: {len(embeddings[0])} dimensions\n")

        return chunks