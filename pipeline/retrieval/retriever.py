"""Retrieval wiring: Chroma metadata-filtered retrieval + source-grounded answer."""

from platform_layer.config.settings import RETRIEVER_K, MAX_CONTEXT_CHUNKS, USE_RERANKER
from pipeline.indexing.vector_store import retrieve_relevant_chunks
from pipeline.retrieval.context_builder import build_context
from pipeline.retrieval.reranker import rerank_chunks


def _retrieve_and_rerank(question: str, document_ids, top_k: int, max_chunks: int):
    """Retrieve top_k chunks, then (optionally) cross-encoder-rerank them so the
    best max_chunks survive into the context."""
    docs = retrieve_relevant_chunks(question, document_ids, top_k)
    if USE_RERANKER:
        docs = rerank_chunks(question, docs, top_n=max_chunks)
    return docs


def answer_with_retrieval(
    answer_chain,
    question: str,
    document_ids,
    top_k: int = RETRIEVER_K,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    summary: str = "",
    history: str = "",
    selected_documents: str = "",
):
    """Retrieve evidence for the selected documents and answer with citations.

    Retrieves up to top_k chunks, includes at most max_chunks of them as labeled
    evidence, and asks the model to produce a grounded answer with sources.
    Conversation memory is passed only to interpret the question. Returns
    (answer, used_docs) so the caller can persist the chunks that were used.
    """
    docs = _retrieve_and_rerank(question, document_ids, top_k, max_chunks)
    context, used_docs = build_context(docs, max_chunks)
    answer = answer_chain.invoke(
        {
            "context": context,
            "input": question,
            "summary": summary or "(none)",
            "history": history or "(none)",
            "selected_documents": selected_documents or "(none)",
        }
    )
    return answer, used_docs


async def stream_answer_with_retrieval(
    answer_chain,
    question: str,
    document_ids,
    top_k: int = RETRIEVER_K,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    summary: str = "",
    history: str = "",
    selected_documents: str = "",
):
    """Same retrieval/grounding as answer_with_retrieval, but streams the
    generation step. Retrieval happens eagerly, so used_docs is known before
    any tokens are yielded. Returns (token_aiter, used_docs)."""
    docs = _retrieve_and_rerank(question, document_ids, top_k, max_chunks)
    context, used_docs = build_context(docs, max_chunks)
    token_aiter = answer_chain.astream(
        {
            "context": context,
            "input": question,
            "summary": summary or "(none)",
            "history": history or "(none)",
            "selected_documents": selected_documents or "(none)",
        }
    )
    return token_aiter, used_docs
