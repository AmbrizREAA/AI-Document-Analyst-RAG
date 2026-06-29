"""ChromaDB-backed vector store.

A single persistent Chroma collection holds the chunks of every processed
document. Each chunk carries rich metadata (document_id, page number, content
hash, ...) so retrieval can filter by document and skip logically-deleted ones.

Embeddings come from the existing HuggingFace / SentenceTransformer model; we
compute them ourselves and hand the vectors to Chroma, so no pickle-based
deserialization is involved (unlike the old FAISS index).
"""

import hashlib

import chromadb
from langchain_core.documents import Document

from config.settings import CHROMA_DIR, CHROMA_COLLECTION, RETRIEVER_K
from indexing.embeddings import get_embeddings

# Lazily-created singletons so the embedding model and Chroma client are each
# built once per process.
_client = None
_collection = None
_embedder = None

# Metadata marker for soft-deleted documents. Retrieval only ever returns
# chunks with status == STATUS_PROCESSED.
STATUS_PROCESSED = "processed"
STATUS_DELETED = "deleted"

# Per-chunk positional metadata we carry through when a loader provides it.
_OPTIONAL_FIELDS = (
    "page_number",
    "slide_number",
    "sheet_name",
    "row_start",
    "row_end",
)


class NoDocumentSelectedError(ValueError):
    """Raised when retrieval is attempted without selecting any document."""


def get_collection():
    """Return the persistent Chroma collection, creating it on first use."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_embedder():
    """Return the shared embedding model, loading it once."""
    global _embedder
    if _embedder is None:
        _embedder = get_embeddings()
    return _embedder


def _content_hash(text: str) -> str:
    """Stable SHA-256 hash of chunk text, used for dedup / change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_metadata(meta: dict) -> dict:
    """Drop None values and keep only Chroma-compatible scalar types."""
    cleaned = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def document_exists(document_id: str) -> bool:
    """Return True if the document already has processed chunks stored."""
    if not document_id:
        return False
    collection = get_collection()
    existing = collection.get(
        where={
            "$and": [
                {"document_id": {"$eq": document_id}},
                {"status": {"$eq": STATUS_PROCESSED}},
            ]
        },
        limit=1,
        include=[],
    )
    return bool(existing["ids"])


def build_chunk_records(document_id: str, chunks, metadata: dict) -> list[dict]:
    """Build per-chunk records shared by ChromaDB and the PostgreSQL metadata store.

    Each record holds the chunk id, raw text, content hash and a cleaned
    metadata dict (document_id, status, content_hash, positional fields, plus the
    per-document base fields). Computing these once keeps both stores consistent.
    """
    base = _clean_metadata(metadata or {})
    records = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        chunk_id = f"{document_id}:{i}"
        content_hash = _content_hash(text)

        meta = dict(base)
        meta["document_id"] = document_id
        meta["chunk_id"] = chunk_id
        meta["status"] = STATUS_PROCESSED
        meta["content_hash"] = content_hash

        # PyMuPDF stores a 0-indexed "page"; expose it as a 1-indexed page_number.
        page = chunk.metadata.get("page")
        if page is not None:
            meta["page_number"] = int(page) + 1

        # Carry through any other positional fields a future loader may set.
        for field in _OPTIONAL_FIELDS:
            if field != "page_number" and chunk.metadata.get(field) is not None:
                meta[field] = chunk.metadata[field]

        records.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "content_hash": content_hash,
                "metadata": _clean_metadata(meta),
            }
        )
    return records


def add_documents_to_vector_store(document_id: str, chunks, metadata: dict) -> int:
    """Embed and upsert a document's chunks into the Chroma collection.

    `metadata` holds the per-document base fields (original_filename,
    safe_filename, file_type). Upsert means reprocessing the same document
    replaces its chunks instead of duplicating them. Returns the chunk count.
    """
    if not chunks:
        return 0

    collection = get_collection()
    embedder = get_embedder()

    records = build_chunk_records(document_id, chunks, metadata)
    texts = [rec["text"] for rec in records]
    embeddings = embedder.embed_documents(texts)

    collection.upsert(
        ids=[rec["chunk_id"] for rec in records],
        documents=texts,
        embeddings=embeddings,
        metadatas=[rec["metadata"] for rec in records],
    )
    return len(records)


def retrieve_relevant_chunks(query: str, selected_document_ids, top_k: int = RETRIEVER_K):
    """Return the top-k chunks for a query, restricted to selected documents.

    Filtering happens inside Chroma: results are limited to the given
    document_ids AND to processed (non-deleted) chunks. Raises
    NoDocumentSelectedError when no document is selected, rather than silently
    searching the entire collection.
    """
    if not selected_document_ids:
        raise NoDocumentSelectedError(
            "Please select or process a document before asking a question."
        )

    collection = get_collection()
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)

    where = {
        "$and": [
            {"document_id": {"$in": list(selected_document_ids)}},
            {"status": {"$eq": STATUS_PROCESSED}},
        ]
    }

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas"],
    )

    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(documents[0], metadatas[0])
    ]


def list_documents() -> list[dict]:
    """List processed documents as {document_id, original_filename} dicts."""
    collection = get_collection()
    records = collection.get(
        where={"status": {"$eq": STATUS_PROCESSED}},
        include=["metadatas"],
    )
    seen = {}
    for meta in records["metadatas"]:
        doc_id = meta.get("document_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = meta.get("original_filename", doc_id)
    return [
        {"document_id": doc_id, "original_filename": name}
        for doc_id, name in sorted(seen.items())
    ]


def delete_document(document_id: str, hard: bool = False) -> None:
    """Remove a document from retrieval.

    Soft delete (default) flips each chunk's status to "deleted" so it can no
    longer be retrieved while keeping the data for auditing. `hard=True`
    physically removes the chunks from the collection.
    """
    if not document_id:
        return
    collection = get_collection()

    if hard:
        collection.delete(where={"document_id": {"$eq": document_id}})
        return

    records = collection.get(
        where={"document_id": {"$eq": document_id}},
        include=["metadatas"],
    )
    ids = records["ids"]
    if not ids:
        return
    updated = []
    for meta in records["metadatas"]:
        new_meta = dict(meta)
        new_meta["status"] = STATUS_DELETED
        updated.append(new_meta)
    collection.update(ids=ids, metadatas=updated)
