"""Split documents into chunks, with a per-format strategy.

- Free-text formats (PDF/DOCX/TXT/MD/JSON): character-based recursive splitting
  with overlap.
- Structured formats (PPTX/XLSX/CSV): the loader already emits one logical unit
  per Document (a slide, or a row block with its column header), so we keep those
  as-is and only split a unit that exceeds MAX_CHUNK_CHARS — preserving metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNK_CHARS

# Formats whose Documents should be split by character count.
TEXT_SPLIT_FORMATS = {"pdf", "docx", "txt", "md", "json"}


def _splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)


def split_documents(documents):
    """Character-based recursive splitting for free-text documents."""
    return _splitter(CHUNK_SIZE).split_documents(documents)


def chunk_documents(documents, file_type: str):
    """Chunk documents according to their format.

    Text formats are recursively split; structured formats keep one chunk per
    logical unit (slide / row block) unless a unit is oversized.
    """
    if file_type in TEXT_SPLIT_FORMATS:
        return split_documents(documents)

    big_splitter = _splitter(MAX_CHUNK_CHARS)
    chunks = []
    for doc in documents:
        if len(doc.page_content) > MAX_CHUNK_CHARS:
            chunks.extend(big_splitter.split_documents([doc]))
        else:
            chunks.append(doc)
    return chunks
