# 🩺 MediRAG

MediRAG is a local Retrieval-Augmented Generation (RAG) application for asking
questions about medical documents. It finds relevant excerpts in an on-device
vector index and asks a Groq-hosted language model to answer from that evidence.

> [!WARNING]
> MediRAG may produce incomplete or incorrect answers. It is not a medical
> device and must not replace diagnosis, treatment, or professional medical
> advice. Verify important information against the original document.

## Features

- PDF, DOCX, TXT, and Markdown ingestion
- Multilingual sentence-transformer embeddings
- Persistent cosine-similarity search with ChromaDB
- Groq answer generation with document and page references
- English, Azerbaijani, and Russian response controls
- Retrieval relevance scoring and weak-match rejection
- Responsive Gradio interface with upload status and index controls
- Safe batch replacement: an invalid upload does not erase a working index
- Configurable retrieval, chunking, file-size, and server settings

The displayed score measures retrieval relevance. It is not a guarantee that an
answer is medically correct.

## Architecture

```text
Document -> Loader -> Token chunker -> Embedder -> ChromaDB
                                                  |
Question -> Optional query rewrite -> Retriever --+
                                      |
                         Context + question -> Groq LLM -> Answer + references
```

## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access on first run to download the embedding model

## Setup

```bash
git clone https://github.com/Fidan6557/medirag.git
cd medirag
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies and create the local environment file:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp` if
`cp` is unavailable. Set `GROQ_API_KEY` in `.env`, then start the app:

```bash
python app.py
```

Open the local URL printed by Gradio, then:

1. Upload one or more supported documents.
2. Choose the answer language and select **Build knowledge base**.
3. Ask focused questions and verify the cited page in the original file.

Successfully processed uploads replace the current local knowledge base. Files
that cannot be processed are skipped, and a batch with no valid documents leaves
the existing knowledge base unchanged.

## Configuration

The following values can be set in `.env`:

| Variable | Required | Default |
|---|---:|---|
| `GROQ_API_KEY` | Yes | — |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL` | No | `paraphrase-multilingual-MiniLM-L12-v2` |
| `CHROMA_PERSIST_DIR` | No | `<project>/chroma_db` |
| `COLLECTION_NAME` | No | `medirag_docs` |
| `CHUNK_SIZE` | No | `512` |
| `CHUNK_OVERLAP` | No | `50` |
| `TOP_K` | No | `3` |
| `CONFIDENCE_THRESHOLD` | No | `0.30` |
| `MAX_FILE_SIZE_MB` | No | `25` |
| `SERVER_NAME` | No | `127.0.0.1` |
| `SERVER_PORT` | No | `7860` |
| `GRADIO_SHARE` | No | `false` |

Local documents, the ChromaDB database, `.env`, virtual environments, and
Python caches are excluded by `.gitignore`.

## Tests

The unit tests use lightweight fakes and do not call Groq or download an
embedding model. Install the development tools and run all checks with:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -v
python -m compileall -q app.py config.py src tests scripts
```

Optional integration checks load the real embedding model. The full-pipeline
check also expects `data/raw/data_for_medirag.pdf` and a working Groq API key:

```bash
python -m scripts.smoke_vectorstore
python -m scripts.smoke_pipeline
```

## Project layout

```text
medirag/
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── generator.py
│   ├── presentation.py
│   └── pipeline.py
├── tests/
├── scripts/
├── app.py
├── config.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Privacy

Document text is stored locally in ChromaDB. Retrieved excerpts and questions
are sent to Groq for answer generation. The application treats document text as
untrusted reference material, but this is not a complete security boundary. Do
not upload sensitive patient data unless your use complies with the applicable
privacy, security, and contractual requirements.

## License

MIT — see [LICENSE](LICENSE).
