"""ChromaDB integration tests: collection init, add, filtered retrieval, delete.

These need a working local ChromaDB and the embedding model (downloaded on first
run), so they are marked `chroma` and are slower. They do not require PostgreSQL.
"""

import os

import pytest
from langchain_core.documents import Document

from platform_layer.config.settings import CHROMA_DIR
from pipeline.indexing.vector_store import (
    get_collection,
    add_documents_to_vector_store,
    retrieve_relevant_chunks,
    document_exists,
    delete_document,
    NoDocumentSelectedError,
)

pytestmark = pytest.mark.chroma


@pytest.fixture
def indexed_doc(unique_id):
    """Add a small document to Chroma and hard-delete it afterwards."""
    doc_id = f"test_chroma_{unique_id}"
    docs = [
        Document(page_content="The capital of Atlantis is Marisol.",
                 metadata={"page": 0}),
        Document(page_content="The lead engineer is Priya Nandkumar.",
                 metadata={"page": 0}),
    ]
    base_metadata = {"original_filename": "atlantis.pdf",
                     "safe_filename": "atlantis.pdf", "file_type": "pdf"}
    add_documents_to_vector_store(doc_id, docs, base_metadata)
    yield doc_id
    delete_document(doc_id, hard=True)


def test_chroma_directory_initializes():
    collection = get_collection()
    assert collection is not None
    assert os.path.isdir(CHROMA_DIR)


def test_add_and_document_exists(indexed_doc):
    assert document_exists(indexed_doc) is True


def test_retrieval_filters_by_selected_document_ids(indexed_doc):
    hits = retrieve_relevant_chunks("capital of Atlantis", [indexed_doc], top_k=3)
    assert hits, "expected at least one chunk for the indexed document"
    assert all(h.metadata.get("document_id") == indexed_doc for h in hits)


def test_unselected_documents_are_excluded(indexed_doc):
    # Querying a different (non-existent) document id returns nothing.
    hits = retrieve_relevant_chunks("capital of Atlantis", ["some_other_doc"], top_k=3)
    assert hits == []


def test_deleted_document_excluded_from_retrieval(indexed_doc):
    delete_document(indexed_doc)  # soft delete (status -> deleted)
    hits = retrieve_relevant_chunks("capital of Atlantis", [indexed_doc], top_k=3)
    assert hits == []


def test_empty_selection_raises_controlled_error():
    with pytest.raises(NoDocumentSelectedError):
        retrieve_relevant_chunks("anything", [], top_k=3)
