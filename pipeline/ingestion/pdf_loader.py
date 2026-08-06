"""PDF ingestion: load a PDF file and clean up extraction artifacts."""

import re

from langchain_community.document_loaders import PyMuPDFLoader


def _normalize_text(text: str) -> str:
    """Clean up PDF extraction artifacts before chunking/embedding.

    Real-world PDFs (especially those from design tools) often extract with
    erratic whitespace: words split across lines, stray blank lines, and runs of
    multiple spaces. Garbled spacing wrecks embedding quality and retrieval, so we
    collapse all whitespace runs to single spaces and trim. This does not repair
    per-character spacing; PyMuPDF avoids that artifact at the
    source, which PyPDFLoader did not.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(file_path: str):
    """Load a PDF into LangChain documents with normalized page text."""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.page_content = _normalize_text(doc.page_content)
    return documents
