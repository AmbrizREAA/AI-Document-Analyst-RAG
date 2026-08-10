"""Tests for the Chainlit upload path.

Chainlit persists spontaneous uploads as `.files/<session>/<uuid>`, often
without the original extension. The pipeline derives the file type from the
filename, so an extension-less path is rejected as "Unsupported file type".
app.py (the Chainlit frontend at the project root) stages each upload under
its original name (element.name) before processing; these tests lock that
behavior in.
"""

import os
import types
import uuid

import pytest

from platform_layer.security.path_safety import has_allowed_extension, file_type_of


def _fake_element(path: str, name: str):
    """Minimal stand-in for a Chainlit File element (path + original name)."""
    return types.SimpleNamespace(path=path, name=name)


def _write_chainlit_style_upload(tmp_path, content: bytes, name: str):
    """Simulate Chainlit's persisted upload: a uuid filename, no extension."""
    upload = tmp_path / uuid.uuid4().hex  # no extension, like .files/<session>/<uuid>
    upload.write_bytes(content)
    return _fake_element(str(upload), name)


# --- Pure unit: the bug's premise --------------------------------------------

def test_extensionless_chainlit_path_is_rejected_by_pipeline():
    """A uuid-only filename (what Chainlit persists) has no usable type."""
    uuid_name = "48bb2eac-200b-4eee-b283-dcabab52a92d"
    assert not has_allowed_extension(uuid_name)
    assert file_type_of(uuid_name) == ""


def test_original_name_restores_file_type():
    """The original filename carried by element.name routes to a loader."""
    assert has_allowed_extension("quarterly_report.txt")
    assert file_type_of("quarterly_report.txt") == "txt"


# --- Integration: full ingest of a Chainlit-style upload ---------------------

def test_chainlit_upload_is_processed_end_to_end(tmp_path, db, cleanup_registry):
    """A uuid-named, extension-less upload must ingest once staged by its
    original name — this is the exact flow _handle_uploads() performs."""
    from app import _stage_upload, ai_system

    content = (
        "Quarterly report: revenue grew 12% year over year. "
        "The retention rate for enterprise clients reached 94%."
    ).encode("utf-8")
    element = _write_chainlit_style_upload(tmp_path, content, "quarterly_report.txt")

    staged = _stage_upload(element)
    try:
        # Staging restores the original filename and the bytes.
        assert os.path.basename(staged) == "quarterly_report.txt"
        with open(staged, "rb") as fh:
            assert fh.read() == content

        status, doc_id = ai_system.process_document(staged)
        assert doc_id is not None, f"upload not processed: {status}"
        cleanup_registry["documents"].add(doc_id)
        assert "processed successfully" in status
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(staged), ignore_errors=True)
