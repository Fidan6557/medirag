"""
chunker.py — Text Splitting Module

Splits raw text from loader into small chunks.
Why is chunking necessary?
  - LLM's context window is limited
  - Smaller pieces are retrieved more accurately
  - Overlap prevents meaning loss
"""

from typing import List, Dict
import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP


def get_tokenizer():
    """
    tiktoken — OpenAI's tokenizer.
    Used to split text into tokens.
    Token ~ word fragment. "MediRAG" = ["Med", "iR", "AG"] could be 3 tokens.
    """
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, source_metadata: Dict) -> List[Dict]:
    """
    Splits text into chunks of CHUNK_SIZE tokens.
    Adds metadata to each chunk.

    How overlap works:
    
    [----chunk 1----]
                [----chunk 2----]
                [overlap]
    
    Without overlap, meaning can be lost:
    "...patient's blood pressure | 120/80 was..."
     chunk 1 ends here   chunk 2 starts here → meaning lost
    """
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0")
    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE

        # Convert tokens back to text
        chunk_tokens = tokens[start:end]
        chunk_text_str = tokenizer.decode(chunk_tokens)

        if chunk_text_str.strip():    # Skip empty chunks
            chunks.append({
                "text": chunk_text_str,
                "metadata": {
                    **source_metadata,          # metadata from loader
                    "chunk_index": chunk_index  # which chunk of this document
                }
            })
            chunk_index += 1

        # Next chunk goes back by overlap amount
        start = end - CHUNK_OVERLAP

    return chunks


def chunk_documents(pages: List[Dict]) -> List[Dict]:
    """
    Splits all pages from loader into chunks.
    Main function to be used in this pipeline.

    Input:  output from load_directory() — List[Dict]
    Output: chunked List[Dict] — ready for embedder
    """
    all_chunks = []

    print(f"Chunking {len(pages)} pages...\n")

    for page in pages:
        chunks = chunk_text(page["text"], page["metadata"])
        all_chunks.extend(chunks)

    print(f"Total {len(all_chunks)} chunks created.")
    print(f"   Average chunk: ~{CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP} tokens\n")

    return all_chunks
