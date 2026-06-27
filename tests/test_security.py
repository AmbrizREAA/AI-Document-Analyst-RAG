"""Unit tests for security utilities (no database or network required)."""

import os

import pytest

from security.path_safety import (
    sanitize_filename,
    has_allowed_extension,
    is_within_size_limit,
    file_type_of,
    safe_index_path,
    VECTOR_STORES_DIR,
)
from config.settings import MAX_FILE_SIZE_BYTES


# --- filename sanitization -------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("report.pdf", "report.pdf"),
    ("my report (final).pdf", "my_report__final_.pdf"),
    ("../../etc/passwd", "passwd"),
    ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe"),
    ("/abs/path/data.csv", "data.csv"),
])
def test_sanitize_filename_strips_paths_and_unsafe_chars(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_never_contains_separators_or_traversal():
    for bad in ["../../x", "a/b/c", "..", ".", "", None]:
        cleaned = sanitize_filename(bad)
        assert "/" not in cleaned and "\\" not in cleaned
        assert cleaned not in ("..", ".")


# --- blocked dangerous filenames -------------------------------------------

def test_dangerous_names_collapse_to_safe_or_empty():
    # Pure traversal tokens leave nothing usable.
    assert sanitize_filename("..") == ""
    assert sanitize_filename("../..") == ""
    # A leading-dot name keeps only the safe remainder.
    assert sanitize_filename(".env") == "env"


# --- allowed extension validation ------------------------------------------

@pytest.mark.parametrize("name", [
    "a.pdf", "a.docx", "a.pptx", "a.xlsx", "a.csv", "a.txt", "a.md", "a.json",
    "A.PDF", "Mixed.Case.Md",
])
def test_allowed_extensions_accepted(name):
    assert has_allowed_extension(name) is True


@pytest.mark.parametrize("name", ["a.exe", "a.sh", "a.zip", "a", "a.pdf.exe", ""])
def test_disallowed_extensions_rejected(name):
    assert has_allowed_extension(name) is False


def test_file_type_of():
    assert file_type_of("Report.PDF") == "pdf"
    assert file_type_of("data.csv") == "csv"
    assert file_type_of("noext") == ""


# --- path traversal prevention / safe path creation ------------------------

def test_safe_index_path_allows_simple_name():
    resolved = safe_index_path("my_doc_faiss")
    assert resolved is not None
    # Stays inside the configured root.
    assert os.path.realpath(resolved).startswith(VECTOR_STORES_DIR)


@pytest.mark.parametrize("bad", [
    "../escape", "..\\escape", "a/b", "a\\b", "..", ".", "", None,
    "/etc/passwd", "C:\\Windows",
])
def test_safe_index_path_blocks_traversal(bad):
    assert safe_index_path(bad) is None


# --- upload size limit -----------------------------------------------------

def test_is_within_size_limit_true_for_small_file(tmp_path):
    f = tmp_path / "small.txt"
    f.write_text("hello world")
    assert is_within_size_limit(str(f)) is True


def test_is_within_size_limit_false_for_oversize(tmp_path, monkeypatch):
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 1024)
    # Pretend the limit is tiny so we don't have to write 25 MB.
    monkeypatch.setattr("security.path_safety.MAX_FILE_SIZE_BYTES", 10)
    assert is_within_size_limit(str(f)) is False


def test_is_within_size_limit_false_for_missing_file():
    assert is_within_size_limit("does/not/exist.txt") is False


def test_size_constant_is_positive():
    assert MAX_FILE_SIZE_BYTES > 0
