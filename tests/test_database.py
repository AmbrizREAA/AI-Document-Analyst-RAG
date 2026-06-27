"""PostgreSQL integration tests: schema, document registry, chunks, memory.

These require a live database (DATABASE_URL in .env). They are skipped — not
failed — when PostgreSQL is unavailable. Created rows are cleaned up afterwards.
"""

import uuid

import pytest
from sqlalchemy import inspect, select

from storage.database import (
    get_engine,
    create_document_record,
    update_document_status,
    get_processed_documents,
    get_document_by_hash,
    save_chunks_metadata,
    create_conversation,
    get_conversation,
    update_conversation_selected_documents,
    get_conversation_summary,
    update_conversation_summary,
    save_message,
    get_recent_messages,
    documents,
    chunks,
)

pytestmark = pytest.mark.integration


# --- initialization & schema ----------------------------------------------

def test_required_tables_exist(db):
    names = set(inspect(get_engine()).get_table_names())
    assert {"documents", "chunks", "conversations", "messages"} <= names


@pytest.mark.parametrize("table,expected", [
    ("documents", {"document_id", "original_filename", "safe_filename", "file_hash",
                   "file_type", "upload_date", "status", "chunk_count",
                   "vector_collection", "summary", "error_message"}),
    ("chunks", {"chunk_id", "document_id", "content_hash", "page_number",
                "slide_number", "sheet_name", "row_start", "row_end",
                "chunk_text", "metadata_json", "created_at"}),
    ("conversations", {"conversation_id", "created_at", "updated_at", "title",
                       "selected_document_ids", "summary"}),
    ("messages", {"message_id", "conversation_id", "role", "content",
                  "created_at", "used_document_ids", "retrieved_chunk_ids"}),
])
def test_table_columns_match_schema(db, table, expected):
    cols = {c["name"] for c in inspect(get_engine()).get_columns(table)}
    assert expected <= cols, f"{table} missing: {expected - cols}"


# --- document registry -----------------------------------------------------

def test_document_registry_lifecycle(db, cleanup_registry, unique_id):
    doc_id = f"test_doc_{unique_id}"
    cleanup_registry["documents"].add(doc_id)
    file_hash = f"test_hash_{unique_id}"

    create_document_record(doc_id, "Report.pdf", "Report.pdf", file_hash,
                           "pdf", "processing", "document_chunks")
    rec = get_document_by_hash(file_hash)
    assert rec is not None
    assert rec["document_id"] == doc_id and rec["status"] == "processing"

    update_document_status(doc_id, "processed", chunk_count=3)
    processed_ids = [d["document_id"] for d in get_processed_documents()]
    assert doc_id in processed_ids

    # Logical delete: excluded from the processed list.
    update_document_status(doc_id, "deleted")
    assert doc_id not in [d["document_id"] for d in get_processed_documents()]


def test_failed_status_stores_error_message(db, cleanup_registry, unique_id):
    doc_id = f"test_fail_{unique_id}"
    cleanup_registry["documents"].add(doc_id)
    create_document_record(doc_id, "bad.pdf", "bad.pdf", f"test_hash_{unique_id}",
                           "pdf", "processing")
    update_document_status(doc_id, "failed", error_message="no extractable text")

    with get_engine().connect() as conn:
        row = conn.execute(
            select(documents).where(documents.c.document_id == doc_id)
        ).mappings().first()
    assert row["status"] == "failed"
    assert row["error_message"] == "no extractable text"


# --- chunk metadata --------------------------------------------------------

def test_save_chunks_metadata_links_and_optional_fields(db, cleanup_registry, unique_id):
    doc_id = f"test_chunks_{unique_id}"
    cleanup_registry["documents"].add(doc_id)
    create_document_record(doc_id, "deck.pptx", "deck.pptx",
                           f"test_hash_{unique_id}", "pptx", "processing")

    records = [
        {"chunk_id": f"{doc_id}:0", "content_hash": "c0", "slide_number": 1,
         "chunk_text": "slide one text",
         "metadata_json": {"document_id": doc_id, "slide_number": 1}},
        {"chunk_id": f"{doc_id}:1", "content_hash": "c1", "page_number": 2,
         "sheet_name": None, "row_start": None, "row_end": None,
         "chunk_text": "page two text", "metadata_json": {"k": "v"}},
    ]
    assert save_chunks_metadata(doc_id, records) == 2

    with get_engine().connect() as conn:
        rows = list(conn.execute(
            select(chunks).where(chunks.c.document_id == doc_id)
            .order_by(chunks.c.chunk_id)
        ).mappings())

    assert len(rows) == 2
    assert all(r["document_id"] == doc_id for r in rows)
    # Optional metadata handled (set vs None).
    assert rows[0]["slide_number"] == 1
    assert rows[1]["page_number"] == 2
    assert rows[1]["sheet_name"] is None
    # JSONB round-trips as a dict.
    assert rows[0]["metadata_json"]["slide_number"] == 1
    assert rows[1]["metadata_json"] == {"k": "v"}


# --- conversation memory ---------------------------------------------------

def test_conversation_memory_roundtrip(db, cleanup_registry, unique_id):
    conv_id = f"test_conv_{unique_id}"
    cleanup_registry["conversations"].add(conv_id)

    create_conversation(conv_id, title="Test chat")
    assert get_conversation(conv_id)["conversation_id"] == conv_id
    assert get_conversation_summary(conv_id) == ""

    save_message(uuid.uuid4().hex, conv_id, "user", "What is the capital?",
                 used_document_ids=["d1"])
    save_message(uuid.uuid4().hex, conv_id, "assistant", "Marisol",
                 used_document_ids=["d1"], retrieved_chunk_ids=["d1:0"])

    recent = get_recent_messages(conv_id, limit=6)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"          # chronological order
    assert recent[-1]["role"] == "assistant"
    assert recent[-1]["retrieved_chunk_ids"] == ["d1:0"]   # JSONB list

    update_conversation_summary(conv_id, "User asked about a capital.")
    assert get_conversation_summary(conv_id) == "User asked about a capital."

    update_conversation_selected_documents(conv_id, ["d1", "d2"])
    assert get_conversation(conv_id)["selected_document_ids"] == ["d1", "d2"]


def test_get_recent_messages_respects_limit(db, cleanup_registry, unique_id):
    conv_id = f"test_limit_{unique_id}"
    cleanup_registry["conversations"].add(conv_id)
    create_conversation(conv_id)
    for i in range(8):
        save_message(uuid.uuid4().hex, conv_id,
                     "user" if i % 2 == 0 else "assistant", f"m{i}")

    recent = get_recent_messages(conv_id, limit=6)
    assert len(recent) == 6
    # The most recent message must be present.
    assert "m7" in {m["content"] for m in recent}
