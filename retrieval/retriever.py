"""Retrieval wiring: Chroma metadata-filtered retrieval + source-grounded answer."""

from config.settings import RETRIEVER_K, MAX_CONTEXT_CHUNKS
from indexing.vector_store import retrieve_relevant_chunks
from retrieval.context_builder import build_context


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
    docs = retrieve_relevant_chunks(question, document_ids, top_k)
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
