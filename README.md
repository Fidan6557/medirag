# 🩺 MediRAG — Medical Document Intelligence System

> A production-ready Retrieval-Augmented Generation (RAG) system for medical document analysis, featuring multilingual support, source transparency, and confidence-aware responses.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

##  What is MediRAG?

MediRAG is an intelligent Q&A assistant that answers questions **grounded in your own medical documents** — not from general LLM knowledge. Upload any clinical guideline, research paper, or medical report, and MediRAG will find the most relevant sections and generate a precise, sourced answer.

Unlike standard LLM chatbots, MediRAG:
- **Never hallucinates** — if the answer isn't in the document, it says so
- **Shows its sources** — every answer includes the exact page and passage used
- **Scores its confidence** — so you know how reliable each response is
- **Speaks your language** — supports English, Azerbaijani, and Russian

---

##  Features

| Feature | Description |
|---|---|
|  Multi-format ingestion | PDF, DOCX, TXT, MD |
|  Semantic search | Embedding-based retrieval with ChromaDB |
|  LLM-powered answers | Groq (Llama 3.1) integration |
|  Source highlighting | Exact page + passage for every answer |
|  Confidence scoring | Retrieval similarity score shown per response |
|  Out-of-scope detection | Refuses to answer if context is insufficient |
|  Multilingual | English, Azerbaijani, Russian |

---

##  Architecture

```
[Document] → [Loader] → [Chunker] → [Embedder] → [ChromaDB]
                                                       ↑
[User Query] → [Embedder] → [Retriever] ───────────────┘
                                  ↓
                       [Context + Query] → [Groq LLM]
                                                ↓
                          [Answer + Sources + Confidence]
```

---

##  Project Structure

```
medirag/
├── src/
│   ├── loader.py          # Document ingestion (PDF, DOCX, TXT, MD)
│   ├── chunker.py         # Smart text splitting with overlap
│   ├── embedder.py        # Sentence-transformer embeddings
│   ├── vectorstore.py     # ChromaDB storage & management
│   ├── retriever.py       # Semantic search + confidence scoring
│   └── generator.py       # Groq LLM answer generation
├── data/
│   └── raw/               # Your medical documents go here
├── app.py                 # Gradio UI
├── config.py              # Configuration & API keys
├── requirements.txt
└── README.md
```

---

##  Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector Database | `ChromaDB` |
| LLM | `Groq` — Llama 3.1 8B |
| Document Parsing | `PyMuPDF`, `python-docx` |
| UI | `Gradio` |
| Evaluation | `RAGAS` |

---

##  Getting Started

```bash
git clone https://github.com/Fidan6557/medirag.git
cd medirag
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
python app.py
```

---

##  Development Roadmap

- [x] **Day 1** — Document loading & chunking pipeline
- [x] **Day 2** — Embedding & vector storage (ChromaDB)
- [x] **Day 3** — Retrieval + LLM generation + confidence scoring
- [x] **Day 4** — Gradio UI with multilingual support and source highlighting

---

## License

MIT License
