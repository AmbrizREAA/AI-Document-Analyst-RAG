"""UI-agnostic orchestration that ties the pipeline modules together.

Vectors live in ChromaDB; document/chunk metadata and conversation memory live
in PostgreSQL. The frontend (the Chainlit app at the project root) keeps only
the selected document_ids and the conversation_id in its session state, never
the documents or chat history themselves.
"""

import os
import uuid
import hashlib

from platform_layer.config.settings import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    CHROMA_COLLECTION,
    SUMMARY_UPDATE_EVERY,
)
from platform_layer.security.path_safety import (
    sanitize_filename,
    has_allowed_extension,
    is_within_size_limit,
    file_type_of,
)
from platform_layer.storage.file_manager import safe_document_id
from pipeline.ingestion.loaders import load_document, DocumentLoadError
from pipeline.processing.chunker import chunk_documents
from pipeline.indexing.vector_store import (
    get_embedder,
    add_documents_to_vector_store,
    build_chunk_records,
    delete_document,
)
from pipeline.llm.provider import get_llm, build_answer_chain, summarize_conversation
from platform_layer.storage.database import (
    init_db,
    create_document_record,
    update_document_status,
    get_document_by_hash,
    save_chunks_metadata,
    create_conversation,
    get_recent_messages,
    update_conversation_summary,
    count_messages,
)


def _file_hash(file_path: str) -> str:
    """SHA-256 of the file's bytes, used to detect re-uploads of the same content."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class DocumentReaderAI:
    """
    Core class handling document ingestion (ChromaDB vectors + PostgreSQL
    metadata) and memory-aware RAG responses. Stateless w.r.t. user data: the
    frontend session holds only the selected document_ids and conversation_id.
    """
    def __init__(self):
        print("Loading model...")
        # Fail fast if PostgreSQL is unreachable; warm the embedder and LLM.
        init_db()
        get_embedder()
        self.llm = get_llm()
        self.answer_chain = build_answer_chain(self.llm)

    # --- Ingestion ---------------------------------------------------------

    def process_document(self, file_path: str):
        """Ingest a document into ChromaDB and record its metadata in PostgreSQL.

        Returns (status, document_id) where document_id is the processed document
        (or None if nothing was ingested) so the caller can auto-select it.
        """
        if not file_path:
            return "Please upload a document.", None

        try:
            clean_name = sanitize_filename(file_path)
            if not has_allowed_extension(clean_name):
                allowed = ", ".join(ALLOWED_EXTENSIONS)
                return f"Unsupported file type. Allowed: {allowed}.", None

            new_document_id = safe_document_id(clean_name)
            if new_document_id is None:
                return "Invalid filename. Please rename the file and try again.", None

            file_type = file_type_of(clean_name)

            if not os.path.isfile(file_path):
                return "The uploaded file could not be found on disk.", None

            if not is_within_size_limit(file_path):
                return f"The file is too large. Maximum size is {MAX_FILE_SIZE_MB} MB.", None

            # Same content already processed? Reuse it instead of duplicating.
            file_hash = _file_hash(file_path)
            existing = get_document_by_hash(file_hash)
            if existing and existing["status"] == "processed":
                return "This document was already in storage. It is now selectable.", existing["document_id"]

            print(f"Loading new document ({file_type}).")
            create_document_record(
                document_id=new_document_id,
                original_filename=os.path.basename(file_path),
                safe_filename=clean_name,
                file_hash=file_hash,
                file_type=file_type,
                status="processing",
                vector_collection=CHROMA_COLLECTION,
            )

            try:
                documents = load_document(file_path, file_type)
            except DocumentLoadError as e:
                update_document_status(new_document_id, "failed", error_message=str(e))
                return str(e), None

            if not documents or sum(len(d.page_content) for d in documents) < 20:
                update_document_status(new_document_id, "failed", error_message="no extractable text")
                return (
                    "No readable text could be extracted. The file may be empty, "
                    "image-only, or unsupported.",
                    None,
                )

            chunks = chunk_documents(documents, file_type)
            if not chunks:
                update_document_status(new_document_id, "failed", error_message="no chunks")
                return "No extractable text was found in the document.", None

            base_metadata = {
                "original_filename": os.path.basename(file_path),
                "safe_filename": clean_name,
                "file_type": file_type,
            }

            # Vectors -> ChromaDB; chunk metadata -> PostgreSQL (same records).
            add_documents_to_vector_store(new_document_id, chunks, base_metadata)
            records = build_chunk_records(new_document_id, chunks, base_metadata)
            save_chunks_metadata(new_document_id, [
                {
                    "chunk_id": r["chunk_id"],
                    "content_hash": r["content_hash"],
                    "page_number": r["metadata"].get("page_number"),
                    "slide_number": r["metadata"].get("slide_number"),
                    "sheet_name": r["metadata"].get("sheet_name"),
                    "row_start": r["metadata"].get("row_start"),
                    "row_end": r["metadata"].get("row_end"),
                    "chunk_text": r["text"],
                    "metadata_json": r["metadata"],
                }
                for r in records
            ])
            update_document_status(new_document_id, "processed", chunk_count=len(records))

            return f"{file_type.upper()} document processed successfully. Please ask a question.", new_document_id

        except FileNotFoundError:
            return "The uploaded file could not be found on disk.", None
        except PermissionError:
            return "Permission denied while reading the file.", None
        except Exception as e:
            print(f"process_document error: {e!r}")
            return "An unexpected error occurred while processing the document. Please try a different file.", None

    def delete_documents(self, document_ids: list[str]) -> int:
        """Logically delete documents: mark status=deleted in PostgreSQL and
        soft-delete their chunks in ChromaDB. Conversation messages are kept.
        Returns the number of documents processed.
        """
        if not document_ids:
            return 0
        for doc_id in document_ids:
            update_document_status(doc_id, "deleted")
            delete_document(doc_id)  # Chroma soft delete (status -> deleted)
        return len(document_ids)

    # --- Conversation memory ----------------------------------------------

    @staticmethod
    def _format_history(messages: list[dict]) -> str:
        """Render stored messages as a compact transcript for the prompt."""
        lines = []
        for m in messages:
            speaker = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {m['content']}")
        return "\n".join(lines)

    def _ensure_conversation(self, conversation_id):
        """Return the conversation id, creating a new conversation if needed."""
        if conversation_id:
            return conversation_id
        new_id = uuid.uuid4().hex
        create_conversation(new_id)
        return new_id

    def _maybe_update_summary(self, conversation_id: str) -> None:
        """Refresh the running summary every SUMMARY_UPDATE_EVERY messages."""
        total = count_messages(conversation_id)
        if total == 0 or total % SUMMARY_UPDATE_EVERY != 0:
            return
        recent = get_recent_messages(conversation_id, limit=SUMMARY_UPDATE_EVERY + 4)
        summary = summarize_conversation(self.llm, self._format_history(recent))
        update_conversation_summary(conversation_id, summary)
