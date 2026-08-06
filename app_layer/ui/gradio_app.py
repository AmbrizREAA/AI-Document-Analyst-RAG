"""Gradio frontend and the orchestration that ties the modules together.

Vectors live in ChromaDB; document/chunk metadata and conversation memory live
in PostgreSQL. Per-session gr.State holds only the selected document_id and the
conversation_id, never the documents or chat history themselves.
"""

import os
import uuid
import hashlib

import gradio as gr

from platform_layer.config.settings import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    CHROMA_COLLECTION,
    RECENT_MESSAGES_LIMIT,
    SUMMARY_UPDATE_EVERY,
)
from platform_layer.security.path_safety import (
    sanitize_filename,
    has_allowed_extension,
    is_within_size_limit,
    file_type_of,
)
from platform_layer.storage.file_manager import safe_document_id
from pipeline.ingestion.loaders import load_document, DocumentLoadError
from pipeline.processing.chunker import chunk_documents
from pipeline.indexing.vector_store import (
    get_embedder,
    add_documents_to_vector_store,
    build_chunk_records,
    delete_document,
    NoDocumentSelectedError,
)
from pipeline.llm.provider import get_llm, build_answer_chain, summarize_conversation
from pipeline.retrieval.retriever import answer_with_retrieval
from platform_layer.storage.database import (
    init_db,
    create_document_record,
    update_document_status,
    get_processed_documents,
    get_document_by_hash,
    save_chunks_metadata,
    create_conversation,
    update_conversation_selected_documents,
    get_conversation_summary,
    update_conversation_summary,
    save_message,
    get_recent_messages,
    count_messages,
)


DOC_TABLE_HEADERS = ["original_filename", "file_type", "status", "chunk_count", "document_id"]


def _document_rows() -> list[list]:
    """Table rows describing each processed document, for the info table."""
    return [
        [d["original_filename"], d["file_type"], d["status"], d["chunk_count"], d["document_id"]]
        for d in get_processed_documents()
    ]


def _document_selector_choices() -> list[tuple]:
    """(label, value) choices for the multi-select, value being the document_id."""
    return [
        (f"{d['original_filename']} — {d['chunk_count']} chunks", d["document_id"])
        for d in get_processed_documents()
    ]


def _valid_document_ids() -> set:
    """Set of currently selectable (processed, non-deleted) document ids."""
    return {d["document_id"] for d in get_processed_documents()}


def _file_hash(file_path: str) -> str:
    """SHA-256 of the file's bytes, used to detect re-uploads of the same content."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class DocumentReaderAI:
    """
    Core class handling PDF ingestion (ChromaDB vectors + PostgreSQL metadata)
    and memory-aware RAG responses. Stateless w.r.t. user data: the per-session
    gr.State holds only the selected document_id and conversation_id.
    """
    def __init__(self):
        print("Loading model...")
        # Fail fast if PostgreSQL is unreachable; warm the embedder and LLM.
        init_db()
        get_embedder()
        self.llm = get_llm()
        self.answer_chain = build_answer_chain(self.llm)

    # --- Ingestion ---------------------------------------------------------

    def process_document(self, file_path: str):
        """Ingest a PDF into ChromaDB and record its metadata in PostgreSQL.

        Returns (status, document_id) where document_id is the processed document
        (or None if nothing was ingested) so the caller can auto-select it.
        """
        if not file_path:
            return "Please upload a PDF document.", None

        try:
            clean_name = sanitize_filename(file_path)
            if not has_allowed_extension(clean_name):
                allowed = ", ".join(ALLOWED_EXTENSIONS)
                return f"Unsupported file type. Allowed: {allowed}.", None

            new_document_id = safe_document_id(clean_name)
            if new_document_id is None:
                return "Invalid filename. Please rename the file and try again.", None

            file_type = file_type_of(clean_name)

            if not os.path.isfile(file_path):
                return "The uploaded file could not be found on disk.", None

            if not is_within_size_limit(file_path):
                return f"The file is too large. Maximum size is {MAX_FILE_SIZE_MB} MB.", None

            # Same content already processed? Reuse it instead of duplicating.
            file_hash = _file_hash(file_path)
            existing = get_document_by_hash(file_hash)
            if existing and existing["status"] == "processed":
                return "This document was already in storage. It is now selectable.", existing["document_id"]

            print(f"Loading new document ({file_type}).")
            create_document_record(
                document_id=new_document_id,
                original_filename=os.path.basename(file_path),
                safe_filename=clean_name,
                file_hash=file_hash,
                file_type=file_type,
                status="processing",
                vector_collection=CHROMA_COLLECTION,
            )

            try:
                documents = load_document(file_path, file_type)
            except DocumentLoadError as e:
                update_document_status(new_document_id, "failed", error_message=str(e))
                return str(e), None

            if not documents or sum(len(d.page_content) for d in documents) < 20:
                update_document_status(new_document_id, "failed", error_message="no extractable text")
                return (
                    "No readable text could be extracted. The file may be empty, "
                    "image-only, or unsupported.",
                    None,
                )

            chunks = chunk_documents(documents, file_type)
            if not chunks:
                update_document_status(new_document_id, "failed", error_message="no chunks")
                return "No extractable text was found in the document.", None

            base_metadata = {
                "original_filename": os.path.basename(file_path),
                "safe_filename": clean_name,
                "file_type": file_type,
            }

            # Vectors -> ChromaDB; chunk metadata -> PostgreSQL (same records).
            add_documents_to_vector_store(new_document_id, chunks, base_metadata)
            records = build_chunk_records(new_document_id, chunks, base_metadata)
            save_chunks_metadata(new_document_id, [
                {
                    "chunk_id": r["chunk_id"],
                    "content_hash": r["content_hash"],
                    "page_number": r["metadata"].get("page_number"),
                    "slide_number": r["metadata"].get("slide_number"),
                    "sheet_name": r["metadata"].get("sheet_name"),
                    "row_start": r["metadata"].get("row_start"),
                    "row_end": r["metadata"].get("row_end"),
                    "chunk_text": r["text"],
                    "metadata_json": r["metadata"],
                }
                for r in records
            ])
            update_document_status(new_document_id, "processed", chunk_count=len(records))

            return f"{file_type.upper()} document processed successfully. Please ask a question.", new_document_id

        except FileNotFoundError:
            return "The uploaded file could not be found on disk.", None
        except PermissionError:
            return "Permission denied while reading the file.", None
        except Exception as e:
            print(f"process_document error: {e!r}")
            return "An unexpected error occurred while processing the document. Please try a different file.", None

    def delete_documents(self, document_ids: list[str]) -> int:
        """Logically delete documents: mark status=deleted in PostgreSQL and
        soft-delete their chunks in ChromaDB. Conversation messages are kept.
        Returns the number of documents processed.
        """
        if not document_ids:
            return 0
        for doc_id in document_ids:
            update_document_status(doc_id, "deleted")
            delete_document(doc_id)  # Chroma soft delete (status -> deleted)
        return len(document_ids)

    # --- Conversation memory ----------------------------------------------

    @staticmethod
    def _format_history(messages: list[dict]) -> str:
        """Render stored messages as a compact transcript for the prompt."""
        lines = []
        for m in messages:
            speaker = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {m['content']}")
        return "\n".join(lines)

    def _ensure_conversation(self, conversation_id):
        """Return the conversation id, creating a new conversation if needed."""
        if conversation_id:
            return conversation_id
        new_id = uuid.uuid4().hex
        create_conversation(new_id)
        return new_id

    def _maybe_update_summary(self, conversation_id: str) -> None:
        """Refresh the running summary every SUMMARY_UPDATE_EVERY messages."""
        total = count_messages(conversation_id)
        if total == 0 or total % SUMMARY_UPDATE_EVERY != 0:
            return
        recent = get_recent_messages(conversation_id, limit=SUMMARY_UPDATE_EVERY + 4)
        summary = summarize_conversation(self.llm, self._format_history(recent))
        update_conversation_summary(conversation_id, summary)

    def answer_question(self, question: str, document_ids, conversation_id):
        """Answer a question over the selected documents, with conversation memory.

        `document_ids` is a list of selected document ids. Returns
        (answer, conversation_id) so Gradio keeps the conversation id in state.
        """
        document_ids = document_ids or []
        if not document_ids:
            return "Please select at least one document to chat with.", conversation_id
        if not question or not question.strip():
            return "Please ask a question.", conversation_id

        # Load memory and record the user's turn. Degrade gracefully if a
        # transient DB issue occurs so the user still gets an answer.
        summary, history = "", ""
        try:
            conversation_id = self._ensure_conversation(conversation_id)
            summary = get_conversation_summary(conversation_id)
            history = self._format_history(
                get_recent_messages(conversation_id, RECENT_MESSAGES_LIMIT)
            )
            save_message(
                message_id=uuid.uuid4().hex,
                conversation_id=conversation_id,
                role="user",
                content=question,
                used_document_ids=document_ids,
            )
        except Exception as e:
            print(f"answer_question memory-load error: {e!r}")

        try:
            answer, docs = answer_with_retrieval(
                self.answer_chain,
                question,
                document_ids,
                summary=summary,
                history=history,
                selected_documents=", ".join(document_ids),
            )
        except NoDocumentSelectedError as e:
            return str(e), conversation_id
        except Exception as e:
            print(f"answer_question error: {e!r}")
            return "Sorry, something went wrong while generating the answer. Please try again.", conversation_id

        # Persist the assistant's turn, selection, and (periodically) summary.
        try:
            chunk_ids = [d.metadata.get("chunk_id") for d in docs if d.metadata.get("chunk_id")]
            save_message(
                message_id=uuid.uuid4().hex,
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                used_document_ids=document_ids,
                retrieved_chunk_ids=chunk_ids,
            )
            update_conversation_selected_documents(conversation_id, document_ids)
            self._maybe_update_summary(conversation_id)
        except Exception as e:
            print(f"answer_question persist error: {e!r}")

        return answer, conversation_id


# --- FRONTEND (Gradio) ---
def create_interface():
    ai_system = DocumentReaderAI()


    corporate_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        radius_size=gr.themes.sizes.radius_none
    ).set(

        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        block_border_width="1px",
        block_background_fill="*neutral_50"
    )

    with gr.Blocks(theme=corporate_theme) as interfaz:

        gr.Markdown("## AI Document Analyst Pro")
        gr.Markdown("Submit documents (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON) and ask questions about their content. Powered by Llama 3, LangChain, MarkItDown, ChromaDB, PostgreSQL, and Groq.")

        gr.HTML("<hr>")

        # Per-session state: list of selected document ids, and the conversation id.
        selected_state = gr.State(value=[])
        conv_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                doc_table = gr.Dataframe(
                    headers=DOC_TABLE_HEADERS,
                    value=_document_rows(),
                    label="Processed documents",
                    interactive=False,
                    wrap=True,
                )
                doc_selector = gr.Dropdown(
                    choices=_document_selector_choices(),
                    value=[],
                    multiselect=True,
                    label="Select documents for chat (one, many, or all)",
                    interactive=True,
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh List")
                    delete_btn = gr.Button("Delete Selected", variant="stop")

                gr.HTML("<hr>")
                pdf_input = gr.File(
                    label="Add a document (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON)",
                    file_types=list(ALLOWED_EXTENSIONS),
                )
                process_btn = gr.Button("Process Document", variant="primary")
                status_output = gr.Textbox(label="System Status", interactive=False)

            with gr.Column(scale=2):
                question_input = gr.Textbox(label="Ask something about the selected documents", lines=2)
                answer_btn = gr.Button("Submit Question")
                answer_output = gr.Textbox(label="AI Response", lines=5)


        def process_and_refresh(file_path, selected_ids):
            status, doc_id = ai_system.process_document(file_path)
            selected_ids = list(selected_ids or [])
            # Auto-select the newly processed (or matched) document for convenience.
            if doc_id and doc_id not in selected_ids:
                selected_ids.append(doc_id)
            return (
                status,
                gr.update(value=_document_rows()),
                gr.update(choices=_document_selector_choices(), value=selected_ids),
                selected_ids,
            )

        def refresh_list(selected_ids):
            valid = _valid_document_ids()
            kept = [i for i in (selected_ids or []) if i in valid]
            rows = _document_rows()
            return (
                f"{len(rows)} document(s) available.",
                gr.update(value=rows),
                gr.update(choices=_document_selector_choices(), value=kept),
                kept,
            )

        def on_select(selected_ids):
            selected_ids = list(selected_ids or [])
            if not selected_ids:
                return "No documents selected.", selected_ids
            return f"{len(selected_ids)} document(s) selected for chat.", selected_ids

        def delete_selected(selected_ids):
            selected_ids = list(selected_ids or [])
            if not selected_ids:
                return (
                    "Select at least one document to delete.",
                    gr.update(),
                    gr.update(),
                    selected_ids,
                )
            count = ai_system.delete_documents(selected_ids)
            return (
                f"Deleted {count} document(s). Conversation history is preserved.",
                gr.update(value=_document_rows()),
                gr.update(choices=_document_selector_choices(), value=[]),
                [],
            )

        process_btn.click(
            fn=process_and_refresh,
            inputs=[pdf_input, selected_state],
            outputs=[status_output, doc_table, doc_selector, selected_state],
        )
        refresh_btn.click(
            fn=refresh_list,
            inputs=[selected_state],
            outputs=[status_output, doc_table, doc_selector, selected_state],
        )
        doc_selector.change(
            fn=on_select,
            inputs=[doc_selector],
            outputs=[status_output, selected_state],
        )
        delete_btn.click(
            fn=delete_selected,
            inputs=[selected_state],
            outputs=[status_output, doc_table, doc_selector, selected_state],
        )
        answer_btn.click(
            fn=ai_system.answer_question,
            inputs=[question_input, selected_state, conv_state],
            outputs=[answer_output, conv_state],
        )

    return interfaz
