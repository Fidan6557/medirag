"""
test_day2.py — Day 2 pipeline test
loader → chunker → embedder → vectorstore
"""

from src.loader import load_document
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vectorstore import VectorStore

# 1) Create test text — no real PDF yet
test_chunks = [
    {
        "text": "Paracetamol is used to treat fever and mild to moderate pain.",
        "metadata": {"source": "test.pdf", "page": 1, "format": "pdf"}
    },
    {
        "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug used for pain relief.",
        "metadata": {"source": "test.pdf", "page": 2, "format": "pdf"}
    },
    {
        "text": "Antibiotics are used to treat bacterial infections, not viral infections.",
        "metadata": {"source": "test.pdf", "page": 3, "format": "pdf"}
    }
]

# Add chunk_index 
for i, chunk in enumerate(test_chunks):
    chunk["metadata"]["chunk_index"] = i

# 2) Generate embeddings
embedder = Embedder()
embedded_chunks = embedder.embed_chunks(test_chunks)

# 3) Add to vector store
vs = VectorStore()
vs.clear()
vs.add_chunks(embedded_chunks)

# 4) Ask a question
print("Question: What treats fever?")
query_embedding = embedder.embed_text("What treats fever?")
results = vs.search(query_embedding, top_k=2)

for i, r in enumerate(results):
    print(f"\n--- Result {i} ---")
    print(f"Text: {r['text']}")
    print(f"Score: {r['score']}")
    print(f"Page:  {r['metadata']['page']}")
