# AI Document Analyst — RAG Application

**Ask natural-language questions about your documents and get source-grounded, cited answers.**

Upload PDFs, Office files, spreadsheets, or plain text; the app converts and chunks them, stores embeddings in **ChromaDB**, keeps document/chunk metadata and conversation history in **PostgreSQL**, and answers questions with **Groq** (`openai/gpt-oss-20b`) using retrieval-augmented generation (RAG).

Built and tested on modest hardware (GTX 1060 + 32 GB RAM) with a focus on a clean, modular, portfolio-friendly design.

---

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON
- **Multi-document retrieval** — select one, many, or all processed documents to query together
- **Cross-encoder reranking** — retrieved chunks are re-scored by a small cross-encoder so only the most relevant evidence reaches the LLM (disable with `USE_RERANKER=false`)
- **Document registry** — every upload is tracked in PostgreSQL with status, chunk count, and a unique content hash (so the same file isn't processed twice)
- **Conversation memory** — recent turns and a rolling summary let follow-up questions ("and its budget?") resolve correctly
- **Source-grounded answers with citations** — answers cite the document name and page/slide/sheet they came from, and the model refuses to answer when the evidence is insufficient
- **Prompt-injection resistance** — retrieved document text is treated as *evidence, not instructions*
- **Logical delete** — documents can be removed from retrieval without losing past conversation history
- **Clean Chainlit chat UI** — streaming answers with clickable source references, drag-and-drop uploads, and a graceful startup error if PostgreSQL or the API key is misconfigured

---

## Architecture

The code is organized into small, single-responsibility modules, grouped under three top-level packages:

```text
AI-Document-Analyst-RAG/
├── app.py                          # Entry point: Chainlit app (run with `chainlit run app.py`)
├── chainlit.md                     # Chainlit welcome screen
├── pipeline/                       # The RAG data flow, end to end
│   ├── ingestion/
│   │   ├── pdf_loader.py           # PyMuPDF PDF loading (page metadata)
│   │   └── loaders.py              # Multi-format dispatcher (MarkItDown / pandas / python-pptx)
│   ├── processing/
│   │   └── chunker.py              # Per-format chunking strategy
│   ├── indexing/
│   │   ├── embeddings.py           # HuggingFace / SentenceTransformers embeddings
│   │   └── vector_store.py         # ChromaDB collection: add / retrieve / soft-delete
│   ├── retrieval/
│   │   ├── retriever.py            # Retrieve + rerank + generate
│   │   ├── reranker.py             # Optional cross-encoder re-scoring of retrieved chunks
│   │   └── context_builder.py      # Source-labeled evidence blocks for citations
│   └── llm/
│       ├── provider.py             # Groq client + answer chain
│       └── prompts.py              # Analyst prompt (grounding, citations, injection resistance)
├── platform_layer/                 # Cross-cutting infrastructure
│   ├── config/
│   │   └── settings.py             # Central config + .env loading (the only place secrets are read)
│   ├── storage/
│   │   ├── database.py             # PostgreSQL (SQLAlchemy Core): documents/chunks/conversations/messages
│   │   └── file_manager.py         # Document-id derivation + legacy-FAISS detection
│   └── security/
│       └── path_safety.py          # Filename sanitization, traversal prevention, size/type checks
└── app_layer/                      # User-facing surface
    ├── ui/
    │   └── document_reader.py      # UI-agnostic orchestration (DocumentReaderAI)
    ├── scripts/
    │   ├── smoke_test.py           # One-shot health check (imports, env, DB, Chroma, config)
    │   └── chainlit_socket_smoke.py# Throwaway socket.io check against a running Chainlit server
    └── tests/                      # pytest suite (unit + integration)
```

### Why ChromaDB (instead of FAISS)

The project originally used FAISS. ChromaDB replaced it because it provides:

- **Metadata filtering** — retrieval can be scoped to selected `document_id`s and skip logically-deleted chunks, which is exactly what multi-document selection needs.
- **A persistent collection** with built-in add/update/delete, instead of one on-disk index file per document.
- **No unsafe pickle deserialization** — FAISS persistence relied on Python `pickle` (arbitrary-code-execution risk when loading untrusted indexes). Chroma stores embeddings without that risk.

> The old `vector_stores/` FAISS folders are no longer used. The app can detect them but never loads them; re-upload the source file to ingest it into ChromaDB.

### Re-indexing after an embedding model change

Embeddings from different models are **not compatible**. The app currently embeds with `BAAI/bge-small-en-v1.5` (previously `all-MiniLM-L6-v2`). If your `chroma_db/` folder still holds vectors from an older model, retrieval quality will silently degrade — clear the collection and re-process your documents:

```bash
# Stop the app, then delete the persisted vectors:
rm -rf chroma_db/          # Windows: rmdir /s /q chroma_db
# Restart the app and re-upload/re-process each document.
```

PostgreSQL metadata does not need to change; re-processing the same file updates the existing registry rows.

### Why PostgreSQL

ChromaDB stores vectors; PostgreSQL stores everything relational:

- **Document registry** (`documents`) — filename, content hash, file type, status (`processing` / `processed` / `failed` / `deleted`), chunk count, error messages.
- **Chunk metadata** (`chunks`) — text, content hash, and optional positional metadata (page / slide / sheet / row range) plus a `metadata_json` (JSONB) copy.
- **Conversation memory** (`conversations`, `messages`) — messages, selected documents, retrieved chunk ids, and a rolling summary.

This keeps structured queries, joins, and history in a real database while embeddings stay in the vector store. **Embeddings are not stored in PostgreSQL**, and **pgvector is not used.**

### DBeaver (optional)

[DBeaver](https://dbeaver.io/) is **optional** and only useful if you want to *visually inspect or manage* the PostgreSQL data (browse the `documents`/`messages` tables, run ad-hoc SQL). The application itself depends only on the `DATABASE_URL` connection string — it does not require DBeaver to run.

---

## Tech Stack

- **Python 3.10+**
- **Chainlit** — chat UI (streaming answers, source references, drag-and-drop uploads)
- **Groq** (`openai/gpt-oss-20b`) — LLM inference
- **HuggingFace / Sentence-Transformers** (`BAAI/bge-small-en-v1.5`) — embeddings
- **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L6-v2`) — retrieval reranking
- **ChromaDB** — local persistent vector store
- **PostgreSQL** via **SQLAlchemy Core** + **psycopg (v3)** — metadata & conversation memory
- **MarkItDown**, **pandas/openpyxl**, **python-pptx**, **PyMuPDF** — document conversion
- **LangChain** (text splitting, chains, prompts)

> Not included: Docker, pgvector, and Qdrant are **not** part of this project.

---

## Supported file types

| Format | Extension | Loader | Notes |
|--------|-----------|--------|-------|
| PDF | `.pdf` | PyMuPDF | page numbers preserved |
| Word | `.docx` | MarkItDown | converted to Markdown |
| PowerPoint | `.pptx` | MarkItDown → python-pptx | one chunk per slide; `slide_number` preserved |
| Excel | `.xlsx` | pandas + openpyxl | row blocks per sheet; `sheet_name` + row range preserved |
| CSV | `.csv` | pandas | row blocks; column headers repeated per chunk |
| Text / Markdown / JSON | `.txt` `.md` `.json` | MarkItDown (plain-text fallback) | |

Max upload size defaults to 25 MB (configurable). **OCR is not included** — scanned/image-only documents won't yield text.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/AmbrizREAA/AI-Document-Analyst-RAG.git
cd AI-Document-Analyst-RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
# Linux / macOS
cp .env.example .env
# Windows (PowerShell)
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```bash
GROQ_API_KEY=your_groq_api_key_here          # https://console.groq.com/keys
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/ai_document_analyst
```

Optional overrides (defaults shown in `.env.example`): `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME`, `MAX_UPLOAD_SIZE_MB`, `TOP_K`, `MAX_CONTEXT_CHUNKS`, `USE_RERANKER`.

### 5. PostgreSQL setup

1. Install and start PostgreSQL (any recent version).
2. Make sure the role/password in your `DATABASE_URL` is valid.
3. The database (`ai_document_analyst` by default) and all tables are **created automatically on first run** — you don't need to create them by hand, as long as the role is allowed to create databases. (To pre-create it manually: `createdb ai_document_analyst`.)
4. *(Optional)* Connect DBeaver to the same `DATABASE_URL` if you want to browse the data.

### 6. Run

```bash
python app.py
# or equivalently:
chainlit run app.py    # add -w for auto-reload during development
```

The app validates `GROQ_API_KEY`, connects to PostgreSQL, initializes tables and the ChromaDB collection, then serves the Chainlit chat UI (default: http://localhost:8000). The embedding model downloads on first run. Launch from the project root so Chainlit picks up the local `.chainlit/` config and `chainlit.md` welcome screen.

---

## Usage

1. **Add a document** — drag a supported file into the chat. It's converted, chunked, embedded into ChromaDB, and registered in PostgreSQL. The same file (by content hash) is never processed twice, and newly processed documents are auto-selected for chat.
2. **Select documents for chat** — open the settings (gear) panel and use the multi-select to pick one, many, or all processed documents. The documents table in the chat shows filename, type, status, and chunk count.
3. **Ask a question** — retrieval is restricted to your selected documents and the answer streams in token by token. It is grounded in the retrieved chunks and comes with clickable **source references** (document name + page/slide/sheet) that open the exact chunk text in the side panel. If the evidence is insufficient, the app says so instead of guessing.
4. **Follow-up questions** work — recent turns and a rolling summary are fed back so pronouns/follow-ups resolve.
5. **Delete** — type `/documents` to list processed documents with per-document delete buttons. Deletion is logical: documents leave retrieval, but past conversation messages are preserved.

If no document is selected, the app shows a clear message instead of searching everything.

---

## Testing

Tests use **pytest**. Pure unit tests (security utilities) run anywhere. Integration tests need a live PostgreSQL (read from `DATABASE_URL`) and/or ChromaDB; they are **skipped with a clear message** when those aren't available — never failed.

```bash
pip install pytest
pytest                      # full suite
pytest -m "not integration" # unit tests only (no database needed)
pytest -m integration       # PostgreSQL tests (requires DATABASE_URL)
pytest -m chroma            # ChromaDB tests (loads the embedding model)
```

Test coverage:

- `app_layer/tests/test_security.py` — filename sanitization, path-traversal prevention, allowed-extension validation, upload size limits
- `app_layer/tests/test_database.py` — DB init, table schema, document registry, logical delete, failed status, chunk metadata (incl. JSONB and optional positional fields), conversation memory
- `app_layer/tests/test_chroma.py` — collection init, add, metadata-filtered retrieval, exclusion of deleted/unselected docs, controlled error on empty selection
- `app_layer/tests/test_reranker.py` — model config (Groq/embedding/reranker names), cross-encoder rerank ordering and trimming, retriever integration with the reranker on/off
- `app_layer/tests/test_chainlit_upload.py` — Chainlit's extension-less upload paths are staged under their original filename before ingestion

### Smoke test

A standalone health check that verifies imports, environment variables, the PostgreSQL connection + tables, the ChromaDB collection, and config — **without launching Chainlit or downloading the LLM**. It masks the database password in its output.

```bash
python app_layer/scripts/smoke_test.py
```

---

## Security considerations

- **`.env` is git-ignored** — never commit it. `DATABASE_URL` (which contains your DB password) and `GROQ_API_KEY` must **not** be committed. Errors and logs mask the password.
- **Local data is git-ignored and must not be committed** — `chroma_db/` (embeddings), `vector_stores/` (legacy FAISS), `uploads/`, `processed_documents/`, and local `*.db`/`*.sqlite3` files all contain or derive from user data.
- **Retrieved document content is evidence, not instructions** — the prompt explicitly instructs the model to ignore any instructions embedded inside documents (prompt-injection resistance).
- **Filename & path safety** — uploads are sanitized; path traversal is blocked; file type and size are validated before processing.
- **No unsafe deserialization** — embeddings are stored in ChromaDB, avoiding the pickle-based loading risk of the old FAISS approach.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `GROQ_API_KEY is not set` at startup | Set a real key in `.env` (not the placeholder). |
| `Could not connect to PostgreSQL` | Server not running, or `DATABASE_URL` host/port/credentials wrong. The error message never prints your password. |
| `password authentication failed` | Wrong user/password in `DATABASE_URL`. |
| `database "ai_document_analyst" does not exist` and isn't auto-created | The DB role lacks `CREATEDB`; create it manually with `createdb ai_document_analyst` or grant the privilege. |
| First question is slow | The embedding model (and the reranker, if enabled) downloads/loads on first use. |
| `No readable text could be extracted` | The file is empty or image-only (no OCR). |
| Integration tests skipped | Expected when `DATABASE_URL` is unset or PostgreSQL is unreachable. |

---

## License

Open source under the MIT License — feel free to use, modify, and learn from it.

Made by **Carlos Alejandro Ambriz**. Feedback and suggestions are welcome — and a ⭐ is appreciated.
Actively seeking entry-level opportunities in Data Analyst | IT Business Analyst | AI Applications.
