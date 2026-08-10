"""Standalone smoke check for the PostgreSQL + ChromaDB architecture.

Run from the project root:

    python app_layer/scripts/smoke_test.py

Verifies that imports work, environment variables are loaded, PostgreSQL is
reachable and initialized, the ChromaDB collection opens, and app config/wiring
loads — all WITHOUT launching the Chainlit server or downloading the LLM. Secrets
are never printed: the DATABASE_URL is shown with its password masked.

Exits 0 if every check passes, 1 otherwise.
"""

import os
import sys

# Make the project root importable when run as `python app_layer/scripts/smoke_test.py`.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

PASS = "[ OK ]"
FAIL = "[FAIL]"


def _mask_db_url(url: str) -> str:
    """Show driver/host/database but never the password."""
    try:
        from sqlalchemy.engine import make_url
        u = make_url(url)
        return f"{u.drivername}://{u.username or '?'}:***@{u.host or '?'}:{u.port or '?'}/{u.database or '?'}"
    except Exception:
        return "(unparseable DATABASE_URL)"


def check_imports():
    import platform_layer.config.settings  # noqa: F401
    import platform_layer.security.path_safety  # noqa: F401
    import platform_layer.storage.database  # noqa: F401
    import pipeline.indexing.vector_store  # noqa: F401
    import pipeline.retrieval.retriever  # noqa: F401
    import pipeline.ingestion.loaders  # noqa: F401
    return "core modules import cleanly"


def check_env():
    from platform_layer.config.settings import GROQ_API_KEY, DATABASE_URL
    problems = []
    if not GROQ_API_KEY or GROQ_API_KEY.strip() in ("", "your_groq_api_key_here"):
        problems.append("GROQ_API_KEY missing/placeholder")
    else:
        print(f"        GROQ_API_KEY: set ({len(GROQ_API_KEY)} chars)")
    if not DATABASE_URL:
        problems.append("DATABASE_URL missing")
    else:
        print(f"        DATABASE_URL: {_mask_db_url(DATABASE_URL)}")
    if problems:
        raise RuntimeError("; ".join(problems))
    return "environment variables loaded"


def check_postgres():
    from platform_layer.storage.database import init_db, get_engine
    from sqlalchemy import inspect
    init_db()  # creates DB/tables if needed; raises a clean error if unreachable
    tables = set(inspect(get_engine()).get_table_names())
    required = {"documents", "chunks", "conversations", "messages"}
    missing = required - tables
    if missing:
        raise RuntimeError(f"missing tables: {missing}")
    return f"PostgreSQL reachable; tables present: {sorted(required)}"


def check_chroma():
    from pipeline.indexing.vector_store import get_collection
    from platform_layer.config.settings import CHROMA_DIR, CHROMA_COLLECTION
    collection = get_collection()
    if not os.path.isdir(CHROMA_DIR):
        raise RuntimeError("Chroma persist directory was not created")
    return f"Chroma collection '{CHROMA_COLLECTION}' ready (count={collection.count()})"


def check_config_and_wiring():
    from platform_layer.config import settings
    # App wiring imports without launching Chainlit or loading the model.
    from app_layer.ui.document_reader import DocumentReaderAI  # noqa: F401
    return (
        f"config OK (formats={len(settings.ALLOWED_EXTENSIONS)}, "
        f"max_upload={settings.MAX_FILE_SIZE_MB}MB, top_k={settings.RETRIEVER_K}, "
        f"max_context_chunks={settings.MAX_CONTEXT_CHUNKS})"
    )


CHECKS = [
    ("imports", check_imports),
    ("environment", check_env),
    ("postgresql", check_postgres),
    ("chromadb", check_chroma),
    ("config/wiring", check_config_and_wiring),
]


def main() -> int:
    print("AI Document Analyst — smoke test\n" + "-" * 40)
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"{PASS} {name}: {detail}")
        except Exception as exc:
            failures += 1
            # Print only the exception message/type, never a full secret-bearing trace.
            print(f"{FAIL} {name}: {type(exc).__name__}: {exc}")
    print("-" * 40)
    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
