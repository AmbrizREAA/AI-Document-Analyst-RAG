"""Shared pytest fixtures and helpers.

Integration tests need a live PostgreSQL (via DATABASE_URL in .env). When the
database is unreachable they are skipped with a clear message rather than failed,
so the pure-unit suite (security utilities) always runs anywhere.
"""

import uuid

import pytest

from platform_layer.config.settings import DATABASE_URL


def _db_available() -> tuple[bool, str]:
    """Return (ok, reason). Never includes the password in the reason string."""
    if not DATABASE_URL:
        return False, "DATABASE_URL is not set in .env"
    try:
        from platform_layer.storage.database import init_db
        init_db()
        return True, ""
    except Exception as exc:  # DatabaseUnavailableError or import-time issues
        return False, f"PostgreSQL unavailable: {type(exc).__name__}"


_DB_OK, _DB_REASON = _db_available()


@pytest.fixture(scope="session")
def db():
    """Session fixture that ensures the database/tables exist, else skips."""
    if not _DB_OK:
        pytest.skip(_DB_REASON)
    return True


@pytest.fixture
def unique_id():
    """A short unique suffix for test rows, so runs don't collide."""
    return uuid.uuid4().hex[:10]


@pytest.fixture
def cleanup_registry():
    """Track ids created during a test and hard-delete them afterwards."""
    created = {"documents": set(), "conversations": set()}
    yield created

    from platform_layer.storage.database import get_engine, documents, chunks, conversations, messages
    try:
        from pipeline.indexing.vector_store import delete_document
    except Exception:
        delete_document = None

    engine = get_engine()
    with engine.begin() as conn:
        for conv_id in created["conversations"]:
            conn.execute(messages.delete().where(messages.c.conversation_id == conv_id))
            conn.execute(conversations.delete().where(conversations.c.conversation_id == conv_id))
        for doc_id in created["documents"]:
            conn.execute(chunks.delete().where(chunks.c.document_id == doc_id))
            conn.execute(documents.delete().where(documents.c.document_id == doc_id))
    if delete_document:
        for doc_id in created["documents"]:
            try:
                delete_document(doc_id, hard=True)
            except Exception:
                pass
