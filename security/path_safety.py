"""Security helpers for safe filename and path handling.

Centralizes the rules the app uses to stay safe with user-supplied files and
on-disk vector stores:

* Only allow known-good file types and a bounded file size.
* Sanitize filenames so they can never escape a target directory.
* Resolve vector-store paths strictly under VECTOR_STORES_DIR, rejecting any
  attempt at path traversal (e.g. "..", absolute paths, embedded separators).
"""

import os
import re

from config.settings import (
    VECTOR_STORES_DIR,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)


def sanitize_filename(filename: str) -> str:
    """Reduce a user-supplied filename to a safe base name.

    Strips any directory components and replaces everything that is not an
    alphanumeric, dash, underscore or dot with an underscore. The result can
    never contain a path separator or "..", so it cannot be used for traversal.
    Returns "" when nothing usable remains.
    """
    if not filename or not isinstance(filename, str):
        return ""
    # Drop any directory part the client may have sent (works for / and \).
    base = os.path.basename(filename.replace("\\", "/"))
    # Keep only a conservative character set.
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    # Guard against names that collapse to a traversal token.
    if cleaned in ("", ".", ".."):
        return ""
    return cleaned


def has_allowed_extension(filename: str) -> bool:
    """Return True if the filename ends with an allowed extension."""
    if not filename:
        return False
    return filename.lower().endswith(ALLOWED_EXTENSIONS)


def file_type_of(filename: str) -> str:
    """Return the lowercase file type (extension without the dot), e.g. 'pdf'."""
    if not filename:
        return ""
    return os.path.splitext(filename)[1].lower().lstrip(".")


def is_within_size_limit(file_path: str) -> bool:
    """Return True if the file exists and is within MAX_FILE_SIZE_BYTES."""
    try:
        return os.path.getsize(file_path) <= MAX_FILE_SIZE_BYTES
    except OSError:
        return False


def safe_index_path(store_name: str) -> str | None:
    """Resolve a vector-store folder strictly under VECTOR_STORES_DIR.

    Returns the absolute path only if the resolved location stays inside
    VECTOR_STORES_DIR. Returns None on attempted path traversal (e.g. "..",
    absolute paths, or names containing a path separator).
    """
    if not store_name or not isinstance(store_name, str):
        return None
    if any(sep in store_name for sep in ("/", "\\")) or store_name in (".", ".."):
        return None
    candidate = os.path.realpath(os.path.join(VECTOR_STORES_DIR, store_name))
    root = VECTOR_STORES_DIR
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    return candidate
