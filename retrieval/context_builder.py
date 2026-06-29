"""Build a source-labeled context string from retrieved chunks.

Each chunk is rendered with a numbered, citable header so the LLM can ground its
answer and cite sources by document name and page/slide/sheet location. The raw
chunk text is presented as evidence only — the prompt instructs the model to
treat it as data, never as instructions.
"""


def _source_locator(meta: dict) -> str:
    """Human-readable location within a document (page / slide / sheet+rows)."""
    parts = []
    if meta.get("page_number") is not None:
        parts.append(f"p.{meta['page_number']}")
    if meta.get("slide_number") is not None:
        parts.append(f"slide {meta['slide_number']}")
    if meta.get("sheet_name"):
        sheet = f"sheet {meta['sheet_name']}"
        if meta.get("row_start") is not None and meta.get("row_end") is not None:
            sheet += f" rows {meta['row_start']}-{meta['row_end']}"
        parts.append(sheet)
    elif meta.get("row_start") is not None and meta.get("row_end") is not None:
        parts.append(f"rows {meta['row_start']}-{meta['row_end']}")
    return ", ".join(parts)


def build_context(docs, max_chunks=None):
    """Render retrieved Documents into a labeled context string.

    Returns (context_string, used_docs) where used_docs is the (possibly
    truncated to max_chunks) list actually included, so callers can persist the
    exact chunk ids that informed the answer.
    """
    if max_chunks is not None:
        docs = docs[:max_chunks]
    if not docs:
        return "(no relevant context was found in the selected documents)", []

    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        name = meta.get("original_filename") or meta.get("document_id") or "unknown"
        locator = _source_locator(meta)
        chunk_id = meta.get("chunk_id", "?")
        header = f"[Source {i} — {name}"
        if locator:
            header += f", {locator}"
        header += f", chunk_id={chunk_id}]"
        blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(blocks), docs
