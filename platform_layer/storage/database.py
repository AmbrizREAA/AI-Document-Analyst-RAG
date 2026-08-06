"""PostgreSQL persistence for documents, chunks, conversations and messages.

Embeddings stay in ChromaDB; PostgreSQL holds relational metadata and
conversation memory only. Built on SQLAlchemy Core (no ORM) to keep the table
definitions and queries explicit and readable. The psycopg (v3) driver is used
via the ``postgresql+psycopg://`` URL scheme.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Text,
    Integer,
    TIMESTAMP,
    ForeignKey,
    func,
    select,
    insert,
    update,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert

from platform_layer.config.settings import DATABASE_URL

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("document_id", Text, primary_key=True),
    Column("original_filename", Text, nullable=False),
    Column("safe_filename", Text, nullable=False),
    Column("file_hash", Text, unique=True, nullable=False),
    Column("file_type", Text, nullable=False),
    Column("upload_date", TIMESTAMP, nullable=False),
    Column("status", Text, nullable=False),
    Column("chunk_count", Integer, server_default="0"),
    Column("vector_collection", Text),
    Column("summary", Text),
    Column("error_message", Text),
)

chunks = Table(
    "chunks",
    metadata,
    Column("chunk_id", Text, primary_key=True),
    Column("document_id", Text, ForeignKey("documents.document_id")),
    Column("content_hash", Text, nullable=False),
    Column("page_number", Integer),
    Column("slide_number", Integer),
    Column("sheet_name", Text),
    Column("row_start", Integer),
    Column("row_end", Integer),
    Column("chunk_text", Text, nullable=False),
    Column("metadata_json", JSONB),
    Column("created_at", TIMESTAMP, nullable=False),
)

conversations = Table(
    "conversations",
    metadata,
    Column("conversation_id", Text, primary_key=True),
    Column("created_at", TIMESTAMP, nullable=False),
    Column("updated_at", TIMESTAMP, nullable=False),
    Column("title", Text),
    Column("selected_document_ids", JSONB),
    Column("summary", Text),
)

messages = Table(
    "messages",
    metadata,
    Column("message_id", Text, primary_key=True),
    Column("conversation_id", Text, ForeignKey("conversations.conversation_id")),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    Column("used_document_ids", JSONB),
    Column("retrieved_chunk_ids", JSONB),
)


class DatabaseUnavailableError(RuntimeError):
    """Raised when PostgreSQL cannot be reached or configured."""


_engine = None


def _now():
    return datetime.now(timezone.utc)


def _require_url() -> str:
    if not DATABASE_URL:
        raise DatabaseUnavailableError(
            "DATABASE_URL is not set. Copy .env.example to .env and set it, e.g. "
            "postgresql+psycopg://postgres:password@localhost:5432/ai_document_analyst"
        )
    return DATABASE_URL


def _ensure_database_exists(url: str) -> None:
    """Create the target database if it doesn't exist yet.

    Connects to the maintenance ``postgres`` database to issue CREATE DATABASE.
    Errors here are surfaced without the password (see init_db).
    """
    try:
        probe = create_engine(url)
        with probe.connect():
            pass
        probe.dispose()
        return
    except OperationalError as exc:
        if "does not exist" not in str(exc).lower():
            raise
    target = make_url(url)
    admin_url = target.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        admin_engine.dispose()


def get_engine():
    """Return the lazily-created SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = _require_url()
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine


def init_db() -> None:
    """Create the database (if missing) and all tables (if missing).

    Raises DatabaseUnavailableError with a helpful, secret-free message when
    PostgreSQL cannot be reached so the app can fail fast at startup.
    """
    url = _require_url()
    try:
        _ensure_database_exists(url)
        engine = get_engine()
        metadata.create_all(engine)
    except OperationalError as exc:
        # OperationalError str() can include the connection URL with password;
        # surface only the safe, leading reason.
        reason = str(exc.orig).splitlines()[0] if exc.orig else "connection failed"
        raise DatabaseUnavailableError(
            "Could not connect to PostgreSQL. Is the server running and is "
            f"DATABASE_URL correct? Reason: {reason}"
        ) from None
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(
            f"Database initialization failed: {type(exc).__name__}"
        ) from None


# --- Documents -------------------------------------------------------------

def create_document_record(
    document_id: str,
    original_filename: str,
    safe_filename: str,
    file_hash: str,
    file_type: str,
    status: str,
    vector_collection: str | None = None,
) -> None:
    """Insert a document row, or update it if the document_id already exists."""
    values = {
        "document_id": document_id,
        "original_filename": original_filename,
        "safe_filename": safe_filename,
        "file_hash": file_hash,
        "file_type": file_type,
        "upload_date": _now(),
        "status": status,
        "vector_collection": vector_collection,
    }
    stmt = pg_insert(documents).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["document_id"],
        set_={
            "original_filename": stmt.excluded.original_filename,
            "safe_filename": stmt.excluded.safe_filename,
            "file_hash": stmt.excluded.file_hash,
            "file_type": stmt.excluded.file_type,
            "upload_date": stmt.excluded.upload_date,
            "status": stmt.excluded.status,
            "vector_collection": stmt.excluded.vector_collection,
        },
    )
    with get_engine().begin() as conn:
        conn.execute(stmt)


def update_document_status(
    document_id: str,
    status: str,
    chunk_count: int | None = None,
    summary: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update a document's status and optional progress fields."""
    values = {"status": status}
    if chunk_count is not None:
        values["chunk_count"] = chunk_count
    if summary is not None:
        values["summary"] = summary
    if error_message is not None:
        values["error_message"] = error_message
    with get_engine().begin() as conn:
        conn.execute(
            update(documents).where(documents.c.document_id == document_id).values(**values)
        )


def get_processed_documents() -> list[dict]:
    """Return processed documents as dicts, newest upload first."""
    stmt = (
        select(documents)
        .where(documents.c.status == "processed")
        .order_by(documents.c.upload_date.desc())
    )
    with get_engine().connect() as conn:
        return [dict(row) for row in conn.execute(stmt).mappings()]


def get_document_by_hash(file_hash: str) -> dict | None:
    """Return the document with this file hash, or None."""
    stmt = select(documents).where(documents.c.file_hash == file_hash)
    with get_engine().connect() as conn:
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None


# --- Chunks ----------------------------------------------------------------

def save_chunks_metadata(document_id: str, chunk_records: list[dict]) -> int:
    """Upsert chunk metadata rows for a document.

    Each record must include at least chunk_id, content_hash and chunk_text;
    positional fields (page_number, etc.) and metadata_json are optional.
    """
    if not chunk_records:
        return 0
    now = _now()
    rows = []
    for rec in chunk_records:
        rows.append(
            {
                "chunk_id": rec["chunk_id"],
                "document_id": document_id,
                "content_hash": rec["content_hash"],
                "page_number": rec.get("page_number"),
                "slide_number": rec.get("slide_number"),
                "sheet_name": rec.get("sheet_name"),
                "row_start": rec.get("row_start"),
                "row_end": rec.get("row_end"),
                "chunk_text": rec["chunk_text"],
                "metadata_json": rec.get("metadata_json"),
                "created_at": now,
            }
        )
    stmt = pg_insert(chunks)
    stmt = stmt.on_conflict_do_update(
        index_elements=["chunk_id"],
        set_={
            "content_hash": stmt.excluded.content_hash,
            "page_number": stmt.excluded.page_number,
            "chunk_text": stmt.excluded.chunk_text,
            "metadata_json": stmt.excluded.metadata_json,
            "created_at": stmt.excluded.created_at,
        },
    )
    with get_engine().begin() as conn:
        conn.execute(stmt, rows)
    return len(rows)


# --- Conversations ---------------------------------------------------------

def create_conversation(
    conversation_id: str,
    title: str | None = None,
    selected_document_ids: list[str] | None = None,
) -> None:
    """Create a conversation row (no-op if it already exists)."""
    now = _now()
    stmt = pg_insert(conversations).values(
        conversation_id=conversation_id,
        created_at=now,
        updated_at=now,
        title=title,
        selected_document_ids=selected_document_ids,
    ).on_conflict_do_nothing(index_elements=["conversation_id"])
    with get_engine().begin() as conn:
        conn.execute(stmt)


def get_conversation(conversation_id: str) -> dict | None:
    """Return a conversation row as a dict, or None."""
    stmt = select(conversations).where(conversations.c.conversation_id == conversation_id)
    with get_engine().connect() as conn:
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None


def update_conversation_selected_documents(
    conversation_id: str, selected_document_ids: list[str]
) -> None:
    """Update which documents are currently selected for a conversation."""
    with get_engine().begin() as conn:
        conn.execute(
            update(conversations)
            .where(conversations.c.conversation_id == conversation_id)
            .values(selected_document_ids=selected_document_ids, updated_at=_now())
        )


def get_conversation_summary(conversation_id: str) -> str:
    """Return the stored summary for a conversation, or '' if none."""
    stmt = select(conversations.c.summary).where(
        conversations.c.conversation_id == conversation_id
    )
    with get_engine().connect() as conn:
        result = conn.execute(stmt).scalar_one_or_none()
        return result or ""


def update_conversation_summary(conversation_id: str, summary: str) -> None:
    """Store an updated conversation summary."""
    with get_engine().begin() as conn:
        conn.execute(
            update(conversations)
            .where(conversations.c.conversation_id == conversation_id)
            .values(summary=summary, updated_at=_now())
        )


# --- Messages --------------------------------------------------------------

def save_message(
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    used_document_ids: list[str] | None = None,
    retrieved_chunk_ids: list[str] | None = None,
) -> None:
    """Persist a single chat message and bump the conversation's updated_at."""
    now = _now()
    with get_engine().begin() as conn:
        conn.execute(
            insert(messages).values(
                message_id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                created_at=now,
                used_document_ids=used_document_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        )
        conn.execute(
            update(conversations)
            .where(conversations.c.conversation_id == conversation_id)
            .values(updated_at=now)
        )


def get_recent_messages(conversation_id: str, limit: int = 6) -> list[dict]:
    """Return the most recent messages in chronological (oldest-first) order."""
    stmt = (
        select(messages)
        .where(messages.c.conversation_id == conversation_id)
        .order_by(messages.c.created_at.desc())
        .limit(limit)
    )
    with get_engine().connect() as conn:
        rows = [dict(row) for row in conn.execute(stmt).mappings()]
    rows.reverse()
    return rows


def count_messages(conversation_id: str) -> int:
    """Return the total number of messages in a conversation."""
    stmt = select(func.count()).select_from(messages).where(
        messages.c.conversation_id == conversation_id
    )
    with get_engine().connect() as conn:
        return int(conn.execute(stmt).scalar_one())
