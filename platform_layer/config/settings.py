"""Centralized configuration and environment loading.

All environment variables and tunable constants live here so the rest of the
app imports settings from one place instead of reading os.getenv scattered
around the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Absolute anchor for everything path-like: the project root is two levels up
# from this file (platform_layer/config/settings.py). Anchoring here — instead
# of os.getcwd() — makes the app behave identically no matter which directory
# `chainlit run` is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# --- Secrets / environment ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# PostgreSQL connection string (SQLAlchemy URL). Example:
#   postgresql+psycopg://postgres:password@localhost:5432/ai_document_analyst
# Stores document/chunk metadata and conversation memory (NOT embeddings).
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Models ---
# NOTE: changing EMBEDDING_MODEL_NAME invalidates every vector already stored in
# ChromaDB — clear the collection and re-process all documents afterwards.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0

# --- Storage ---
# Persistent ChromaDB directory. One local collection holds every document's
# chunks; deletion and metadata filtering happen inside Chroma. Override the
# location/name with CHROMA_PERSIST_DIR / CHROMA_COLLECTION_NAME in .env.
# Defaults to <project root>/chroma_db regardless of the launch directory.
CHROMA_DIR = os.path.realpath(
    os.getenv("CHROMA_PERSIST_DIR") or (PROJECT_ROOT / "chroma_db")
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "document_chunks")

# Legacy: root directory of the old FAISS vector stores. No longer used for
# storage; kept only so the app can detect pre-migration folders (see
# storage/file_manager.list_legacy_faiss_stores).
VECTOR_STORES_DIR = os.path.realpath(PROJECT_ROOT / "vector_stores")

# --- Uploads ---
# Supported document formats. Add new extensions here (lowercase, with dot)
# only once the ingestion layer can handle them.
ALLOWED_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".txt",
    ".md",
    ".json",
)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Chunking ---
# Character-based splitting for free-text formats (PDF/DOCX/TXT/MD/JSON). Roughly
# 1000 chars per chunk with overlap; tune here to trade retrieval precision for
# context size. Structured formats are chunked by their own units (see below).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# Safety cap: a single "logical" chunk (one slide / one row-block) larger than
# this many characters is further split so no chunk is unbounded.
MAX_CHUNK_CHARS = 4000
# Spreadsheet/CSV: number of data rows per chunk (column header repeated on each).
ROWS_PER_CHUNK = 40

# --- Retrieval ---
# How many chunks to retrieve from the vector store per question (top_k).
RETRIEVER_K = int(os.getenv("TOP_K", "5"))
# Upper bound on how many of those chunks are actually sent to the LLM as
# evidence. Keeps the prompt bounded and citations manageable.
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "4"))
# Optional cross-encoder reranking: re-scores the retrieved chunks and keeps
# the best MAX_CONTEXT_CHUNKS before the context is built. Small fp32 model,
# safe on modest GPUs/CPU; disable with USE_RERANKER=false if it is too slow.
USE_RERANKER = os.getenv("USE_RERANKER", "true").strip().lower() in ("1", "true", "yes", "on")
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
# Cap on the model's answer length (passed to the provider as max_tokens).
MAX_ANSWER_TOKENS = 512

# --- Conversation memory ---
# How many recent turns to feed back into the prompt for follow-up questions.
RECENT_MESSAGES_LIMIT = 6
# Refresh the running conversation summary every N messages (kept in the 6-10 range).
SUMMARY_UPDATE_EVERY = 6
