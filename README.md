# 🩺 MediRAG — Medical Document Intelligence System

> A production-ready Retrieval-Augmented Generation (RAG) system for medical document analysis, featuring multilingual support, source transparency, and confidence-aware responses.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## What is MediRAG?

MediRAG is an intelligent Q&A assistant that answers questions **grounded in your own medical documents** — not from general LLM knowledge. Upload any clinical guideline, research paper, or medical report, and MediRAG will find the most relevant sections and generate a precise, sourced answer.

Unlike standard LLM chatbots, MediRAG:
- **Never hallucinates** — if the answer isn't in the document, it says so
- **Shows its sources** — every answer includes the exact page and passage used
- **Scores its confidence** — so you know how reliable each response is
- **Speaks your language** — supports English, Azerbaijani, and Russian

---

## Features

| Feature | Description |
|---|---|
|  Multi-format ingestion | PDF, DOCX, TXT, MD |
|  Semantic search | Embedding-based retrieval with ChromaDB |
|  LLM-powered answers | Groq (Llama 3) integration |
|  Source highlighting | Exact page + passage for every answer |
|  Confidence scoring | Retrieval similarity score shown per response |
|  Out-of-scope detection | Refuses to answer if context is insufficient |
|  Multilingual | English, Azerbaijani, Russian |
|  Evaluation dashboard | Precision, recall, faithfulness metrics via RAGAS |

---

##  Architecture

\```
┌─────────────────────────────────────────────────────────┐
│                      MediRAG Pipeline                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Document] ──► [Loader] ──► [Chunker] ──► [Embedder]  │
│                                                  │      │
│                                            [ChromaDB]   │
│                                                  │      │
│  [User Query] ──► [Embedder] ──► [Retriever] ───┘      │
│                                       │                 │
│                              [Context + Query]          │
│                                       │                 │
│                               [Groq LLM]                │
│                                       │                 │
│                    [Answer + Sources + Confidence]      │
└─────────────────────────────────────────────────────────┘
\```

---

##  Project Structure

\```
medirag/
├── src/
│   ├── loader.py         # Document ingestion (PDF, DOCX, TXT, MD)
│   ├── chunker.py        # Smart text splitting with overlap
│   ├── embedder.py       # Sentence-transformer embeddings
│   ├── vectorstore.py    # ChromaDB storage & management
│   ├── retriever.py      # Semantic search + confidence scoring
│   └── generator.py      # Groq LLM answer generation
├── notebooks/
│   ├── 01_eda.ipynb      # Data exploration
│   ├── 02_pipeline.ipynb # End-to-end pipeline demo
│   └── 03_eval.ipynb     # RAGAS evaluation
├── data/
│   └── raw/              # Your medical documents go here
├── app.py                # Streamlit UI
├── config.py             # Configuration & API keys
├── requirements.txt
└── README.md
\```

---

##  Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Database | `ChromaDB` |
| LLM | `Groq` — Llama 3.1 8B |
| Document Parsing | `PyMuPDF`, `python-docx` |
| UI | `Streamlit` |
| Evaluation | `RAGAS` |

---

##  Development Roadmap

- [x] **Day 1** — Document loading & chunking pipeline
- [ ] **Day 2** — Embedding & vector storage (ChromaDB)
- [ ] **Day 3** — Retrieval + LLM generation + confidence scoring
- [ ] **Day 4** — Streamlit UI + evaluation dashboard + deployment

---

##  License

MIT License