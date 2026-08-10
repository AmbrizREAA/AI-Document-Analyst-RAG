"""Unit tests for model configuration and the cross-encoder reranker.

Pure unit tests — no database, no ChromaDB, and no live model downloads: the
cross-encoder is replaced with a stub that returns fixed scores, so the
ordering/trimming logic and the retriever wiring are verified offline.
"""

import pytest
from langchain_core.documents import Document

from platform_layer.config import settings
from pipeline.retrieval import reranker as reranker_module
from pipeline.retrieval import retriever as retriever_module
from pipeline.retrieval.reranker import rerank_chunks


def _doc(chunk_id, text="chunk text"):
    return Document(
        page_content=text,
        metadata={"document_id": "doc", "chunk_id": chunk_id},
    )


class _StubCrossEncoder:
    """CrossEncoder stand-in returning a fixed score per (query, text) pair."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return self._scores


@pytest.fixture
def stub_reranker(monkeypatch):
    """Replace the lazy singleton with a stub whose scores can be set per test."""
    holder = {}

    def _get():
        return holder["model"]

    monkeypatch.setattr(reranker_module, "get_reranker", _get)
    return holder


# --- Model configuration -----------------------------------------------------


def test_llm_model_is_current_groq_model():
    # llama-3.1-8b-instant is deprecated by Groq; gpt-oss-20b replaces it.
    assert settings.LLM_MODEL_NAME == "openai/gpt-oss-20b"
    assert "llama-3.1-8b-instant" not in settings.LLM_MODEL_NAME


def test_embedding_model_is_bge_small():
    assert settings.EMBEDDING_MODEL_NAME == "BAAI/bge-small-en-v1.5"


def test_reranker_defaults_to_enabled_with_expected_model():
    assert settings.USE_RERANKER is True
    assert settings.RERANKER_MODEL_NAME == "cross-encoder/ms-marco-MiniLM-L6-v2"


# --- rerank_chunks ------------------------------------------------------------


def test_rerank_orders_chunks_by_score(stub_reranker):
    docs = [_doc("a"), _doc("b"), _doc("c")]
    stub_reranker["model"] = _StubCrossEncoder([0.1, 0.9, 0.5])

    ranked = rerank_chunks("question", docs)

    assert [d.metadata["chunk_id"] for d in ranked] == ["b", "c", "a"]
    # The model was queried with every (query, text) pair, in input order.
    assert stub_reranker["model"].calls == [
        [("question", "chunk text")] * 3
    ]


def test_rerank_trims_to_top_n(stub_reranker):
    docs = [_doc("a"), _doc("b"), _doc("c")]
    stub_reranker["model"] = _StubCrossEncoder([0.1, 0.9, 0.5])

    ranked = rerank_chunks("question", docs, top_n=2)

    assert [d.metadata["chunk_id"] for d in ranked] == ["b", "c"]


def test_rerank_empty_input_short_circuits(monkeypatch):
    def _boom():
        raise AssertionError("reranker model must not load for empty input")

    monkeypatch.setattr(reranker_module, "get_reranker", _boom)
    assert rerank_chunks("question", []) == []


# --- Retriever wiring ---------------------------------------------------------


def _stub_retrieval(monkeypatch, docs):
    monkeypatch.setattr(
        retriever_module, "retrieve_relevant_chunks",
        lambda question, document_ids, top_k: docs,
    )


def test_answer_with_retrieval_applies_reranker_when_enabled(monkeypatch):
    docs = [_doc("a"), _doc("b"), _doc("c")]
    _stub_retrieval(monkeypatch, docs)
    monkeypatch.setattr(retriever_module, "USE_RERANKER", True)
    # Reranker reverses the order and keeps only the top 2.
    monkeypatch.setattr(
        retriever_module, "rerank_chunks",
        lambda question, docs, top_n=None: list(reversed(docs))[:top_n],
    )

    class _Chain:
        def invoke(self, inputs):
            return inputs["context"]

    answer, used = retriever_module.answer_with_retrieval(
        _Chain(), "question", ["doc"], top_k=3, max_chunks=2
    )

    assert [d.metadata["chunk_id"] for d in used] == ["c", "b"]
    assert "chunk_id=c" in answer
    assert "chunk_id=a" not in answer  # trimmed by the reranker


def test_answer_with_retrieval_skips_reranker_when_disabled(monkeypatch):
    docs = [_doc("a"), _doc("b")]
    _stub_retrieval(monkeypatch, docs)
    monkeypatch.setattr(retriever_module, "USE_RERANKER", False)

    def _boom(*args, **kwargs):
        raise AssertionError("reranker must not run when USE_RERANKER is false")

    monkeypatch.setattr(retriever_module, "rerank_chunks", _boom)

    class _Chain:
        def invoke(self, inputs):
            return inputs["context"]

    answer, used = retriever_module.answer_with_retrieval(
        _Chain(), "question", ["doc"], top_k=2, max_chunks=2
    )

    assert [d.metadata["chunk_id"] for d in used] == ["a", "b"]
