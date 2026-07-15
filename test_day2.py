"""
test_day2.py — Day 2 pipeline test
loader → chunker → embedder → vectorstore
"""

from src.embedder import Embedder
from src.vectorstore import VectorStore


def main():
    test_chunks = [
        {
            "text": "Paracetamol is used to treat fever and mild to moderate pain.",
            "metadata": {"source": "test.pdf", "page": 1, "format": "pdf"},
        },
        {
            "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug used for pain relief.",
            "metadata": {"source": "test.pdf", "page": 2, "format": "pdf"},
        },
        {
            "text": "Antibiotics treat bacterial infections, not viral infections.",
            "metadata": {"source": "test.pdf", "page": 3, "format": "pdf"},
        },
    ]

    for index, chunk in enumerate(test_chunks):
        chunk["metadata"]["chunk_index"] = index

    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(test_chunks)

    vectorstore = VectorStore()
    vectorstore.clear()
    vectorstore.add_chunks(embedded_chunks)

    print("Question: What treats fever?")
    query_embedding = embedder.embed_text("What treats fever?")
    results = vectorstore.search(query_embedding, top_k=2)

    for index, result in enumerate(results):
        print(f"\n--- Result {index} ---")
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']}")
        print(f"Page:  {result['metadata']['page']}")


if __name__ == "__main__":
    main()
