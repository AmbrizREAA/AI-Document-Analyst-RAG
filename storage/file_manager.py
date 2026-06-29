"""Document naming, listing, and a legacy-FAISS compatibility note.

Storage now lives in a single ChromaDB collection (see indexing/vector_store).
Documents are identified by a conservative, filesystem-safe id derived from the
uploaded filename, which also keeps reprocessing the same file idempotent.
"""

import os
import re

from config.settings import VECTOR_STORES_DIR
from indexing.vector_store import list_documents


def safe_document_id(filename: str) -> str | None:
    """Derive a stable, conservative document id from an uploaded filename.

    Strips the file extension and reduces the name to alphanumerics/underscores.
    Returns None if nothing usable remains. The same file always maps to the
    same id, so reprocessing replaces rather than duplicates.
    """
    base_name = os.path.splitext(filename)[0]
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", base_name).strip("_")
    return safe_name or None


def list_existing_documents() -> list[str]:
    """Return processed document ids for the UI dropdown."""
    return [doc["document_id"] for doc in list_documents()]


def list_legacy_faiss_stores() -> list[str]:
    """Detect pre-migration FAISS folders, if any still exist.

    The app no longer reads or writes FAISS indexes. These folders are inert;
    to use such a document again, re-upload the original PDF so it is ingested
    into ChromaDB. This helper exists only so a UI/CLI can surface a one-time
    "these old stores are no longer used" note.
    """
    if not os.path.isdir(VECTOR_STORES_DIR):
        return []
    return sorted(
        d for d in os.listdir(VECTOR_STORES_DIR)
        if os.path.isdir(os.path.join(VECTOR_STORES_DIR, d)) and d.endswith("_faiss")
    )
