# 🩺 MediRAG

MediRAG is a local Retrieval-Augmented Generation (RAG) prototype for asking
questions about medical documents. It retrieves relevant excerpts from uploaded
files and asks a Groq-hosted language model to answer using those excerpts.

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
- Local Gradio interface

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

Open the local URL printed by Gradio, upload one or more supported documents,
select a response language, and process the files before asking questions.
Processing a new upload replaces the current local knowledge base.

## Configuration

The following values can be set in `.env`:

| Variable | Required | Default |
|---|---:|---|
| `GROQ_API_KEY` | Yes | — |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` |
| `CHROMA_PERSIST_DIR` | No | `<project>/chroma_db` |
| `COLLECTION_NAME` | No | `medirag_docs` |

Local documents, the ChromaDB database, `.env`, virtual environments, and
Python caches are excluded by `.gitignore`.

## Tests

The unit tests use lightweight fakes and do not call Groq or download an
embedding model:

```bash
python -m unittest discover -v
python -m compileall -q app.py config.py src tests
```

`test_day2.py` and `test_day3.py` are optional manual integration checks. They
load the real embedding model; `test_day3.py` also expects
`data/raw/data_for_medirag.pdf` and a working Groq API key.

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
│   └── pipeline.py
├── tests/
├── app.py
├── config.py
├── requirements.txt
└── .env.example
```

## Privacy

Document text is stored locally in ChromaDB. Retrieved excerpts and questions
are sent to Groq for answer generation. Do not upload sensitive patient data
unless your use complies with the applicable privacy, security, and contractual
requirements.

## License

MIT — see [LICENSE](LICENSE).
