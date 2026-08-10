"""Optional cross-encoder reranking of retrieved chunks.

Vector similarity (bi-encoder) is fast but coarse; a small cross-encoder
re-scores each (query, chunk) pair jointly and reorders the candidates so the
chunks actually sent to the LLM are the most relevant ones. The model is tiny
(~22M params, fp32), so it runs comfortably on a modest GPU or plain CPU.
"""

from sentence_transformers import CrossEncoder

from platform_layer.config.settings import RERANKER_MODEL_NAME

# Lazily-created singleton so the model is loaded once per process.
_reranker = None


def get_reranker() -> CrossEncoder:
    """Return the shared cross-encoder model, loading it on first use."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank_chunks(query: str, docs, top_n: int = None):
    """Re-score `docs` against `query` and return them best-first.

    When `top_n` is given, only the top_n highest-scoring chunks are kept.
    An empty input is returned unchanged without loading the model.
    """
    if not docs:
        return docs
    scores = get_reranker().predict([(query, doc.page_content) for doc in docs])
    ranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda p: p[0], reverse=True)]
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked
