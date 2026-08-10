"""Chainlit frontend for AI Document Analyst Pro — the app entry point.

Reuses the UI-agnostic DocumentReaderAI orchestration class
(app_layer/ui/document_reader.py); this is a thin UI layer, not a rewrite.
Session state lives in cl.user_session: only the selected document_ids and
the conversation_id, never the documents or chat history themselves.

Run with:  chainlit run app.py -w
Or simply: python app.py   (launches the same Chainlit server)
"""

import asyncio
import os
import shutil
import sys
import tempfile
import uuid

if getattr(asyncio, "_nest_patched", False) and hasattr(asyncio.tasks, "_py_current_task"):
    asyncio.tasks.current_task = asyncio.tasks._py_current_task
    asyncio.tasks.all_tasks = asyncio.tasks._py_all_tasks
    asyncio.current_task = asyncio.tasks._py_current_task
    asyncio.all_tasks = asyncio.tasks._py_all_tasks

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))

import pandas as pd
import chainlit as cl
from chainlit.input_widget import MultiSelect

from app_layer.ui.document_reader import DocumentReaderAI
from platform_layer.config.settings import PROJECT_ROOT, RECENT_MESSAGES_LIMIT
from platform_layer.storage.database import (
    get_processed_documents,
    get_conversation_summary,
    get_recent_messages,
    save_message,
    update_conversation_selected_documents,
)
from pipeline.indexing.vector_store import NoDocumentSelectedError
from pipeline.retrieval.context_builder import _source_locator
from pipeline.retrieval.retriever import stream_answer_with_retrieval

if __name__ != "__main__":
    ai_system = DocumentReaderAI()

from chainlit.config import APP_ROOT as _CHAINLIT_APP_ROOT

if os.path.realpath(_CHAINLIT_APP_ROOT) != os.path.realpath(PROJECT_ROOT):
    print(
        f"WARNING: chainlit is using APP_ROOT={_CHAINLIT_APP_ROOT!r}, so it reads "
        f".chainlit/ and .files/ from there, not from the project root "
        f"{str(PROJECT_ROOT)!r}. Launch from the project root "
        "(chainlit run app.py) or set CHAINLIT_APP_ROOT."
    )


@cl.data_layer
def _disable_chainlit_persistence():
    """Keep Chainlit's built-in persistence off.

    Chainlit auto-activates its own SQL data layer when DATABASE_URL is present
    in the environment — but that variable belongs to this app's own Postgres
    registry/memory in platform_layer/storage/database.py. Returning None here
    takes precedence over the DATABASE_URL auto-detection.
    """
    return None

DOC_TABLE_COLUMNS = ["original_filename", "file_type", "status", "chunk_count", "document_id"]

USAGE_HINT = (
    "Tip: drag a file into the chat to upload it, use the settings (gear) icon "
    "to choose which documents to chat with, then just ask. Type /documents to "
    "manage or delete processed documents."
)


# --- Shared renderers -------------------------------------------------------

def _documents_dataframe() -> pd.DataFrame:
    """Processed documents as a table shown in the chat."""
    return pd.DataFrame(list(get_processed_documents()), columns=DOC_TABLE_COLUMNS)


def _selector_items() -> dict:
    """{label: document_id} for the settings multi-select (MultiSelect API)."""
    return {
        f"{d['original_filename']} — {d['chunk_count']} chunks": d["document_id"]
        for d in get_processed_documents()
    }


async def _send_documents_table(intro: str = "Processed documents:") -> None:
    await cl.Message(
        content=intro,
        elements=[cl.Dataframe(name="documents", data=_documents_dataframe(), display="inline")],
    ).send()


async def _send_chat_settings() -> None:
    """(Re)send the settings panel with the current document choices."""
    items = _selector_items()
    if not items:
        return  # MultiSelect requires at least one item; nothing to select yet.
    selected = cl.user_session.get("selected_document_ids") or []
    initial = [doc_id for doc_id in selected if doc_id in items.values()]
    await cl.ChatSettings(
        inputs=[
            MultiSelect(
                id="documents",
                label="Select documents for chat (one, many, or all)",
                items=items,
                initial=initial,
            )
        ]
    ).send()


# --- Chat lifecycle ---------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("selected_document_ids", [])
    cl.user_session.set("conversation_id", None)
    await _send_documents_table(
        "Welcome to AI Document Analyst Pro. These documents are already processed:"
    )
    await _send_chat_settings()
    await cl.Message(content=USAGE_HINT).send()


@cl.on_settings_update
async def on_settings_update(settings):
    valid_ids = {d["document_id"] for d in get_processed_documents()}
    selected = [i for i in (settings.get("documents") or []) if i in valid_ids]
    cl.user_session.set("selected_document_ids", selected)
    await cl.Message(
        content=(
            f"{len(selected)} document(s) selected for chat."
            if selected
            else "No documents selected."
        )
    ).send()


# --- Uploads ----------------------------------------------------------------

def _stage_upload(element) -> str:
    """Copy an uploaded element to a temp file that keeps the original filename.

    Chainlit persists spontaneous uploads as `.files/<session>/<uuid>`, often
    without the original extension, while the pipeline derives the file type
    from the filename. Staging under element.name restores the extension so
    process_document() can route the file to the right loader. Caller is
    responsible for deleting the returned path (and its temp directory).
    """
    original = os.path.basename((element.name or "").replace("\\", "/")) or "upload"
    tmp_dir = tempfile.mkdtemp(prefix="chainlit_upload_")
    staged = os.path.join(tmp_dir, original)
    shutil.copyfile(element.path, staged)
    return staged


async def _handle_uploads(files) -> None:
    """Process spontaneously uploaded files (drag & drop into the chat)."""
    for element in files:
        try:
            staged = _stage_upload(element)
            try:
                status, doc_id = await cl.make_async(ai_system.process_document)(staged)
            finally:
                shutil.rmtree(os.path.dirname(staged), ignore_errors=True)
        except Exception as e:
            print(f"chainlit upload error ({element.name}): {e!r}")
            await cl.Message(
                content=f"{element.name}: processing failed ({type(e).__name__}: {e})"
            ).send()
            continue
        await cl.Message(content=f"{element.name}: {status}").send()
        # Auto-select the newly processed (or deduplicated) document.
        if doc_id:
            selected = cl.user_session.get("selected_document_ids") or []
            if doc_id not in selected:
                selected.append(doc_id)
                cl.user_session.set("selected_document_ids", selected)
    await _send_documents_table()
    await _send_chat_settings()


# --- /documents command (delete) --------------------------------------------

async def _send_documents_command() -> None:
    docs = get_processed_documents()
    await _send_documents_table()
    if not docs:
        await cl.Message(content="No processed documents to manage yet.").send()
        return
    actions = [
        cl.Action(
            name="delete_doc",
            payload={"document_id": d["document_id"]},
            label=f"Delete {d['original_filename']}",
        )
        for d in docs
    ]
    await cl.Message(content="Delete a document:", actions=actions).send()


@cl.action_callback("delete_doc")
async def on_delete(action: cl.Action):
    doc_id = action.payload["document_id"]
    await cl.make_async(ai_system.delete_documents)([doc_id])
    selected = [
        i for i in (cl.user_session.get("selected_document_ids") or []) if i != doc_id
    ]
    cl.user_session.set("selected_document_ids", selected)
    await action.remove()
    await cl.Message(content="Deleted. Conversation history is preserved.").send()
    await _send_documents_table()
    await _send_chat_settings()


# --- Questions (streaming) ---------------------------------------------------

def _source_elements(used_docs) -> list:
    """Compact source references for the answer.

    display="side" renders each source as a small reference chip beside the
    answer instead of a full-width inline card; clicking it opens the chunk
    text in Chainlit's side panel. Sources are deduplicated by
    (document, page/locator) — multiple chunks from the same page collapse
    into one chip, keeping the order of first appearance.
    """
    elements = []
    seen = set()
    for doc in used_docs:
        meta = doc.metadata or {}
        name = meta.get("original_filename") or meta.get("document_id") or "unknown"
        locator = _source_locator(meta) or "location unknown"
        if (name, locator) in seen:
            continue
        seen.add((name, locator))
        elements.append(
            cl.Text(
                name=f"{name} — {locator}",
                content=doc.page_content,
                display="side",
            )
        )
    return elements


@cl.on_message
async def on_message(message: cl.Message):
    # Spontaneous uploads ride along with the message as file elements.
    uploaded = [e for e in (message.elements or []) if getattr(e, "path", None)]
    if uploaded:
        await _handle_uploads(uploaded)

    text = (message.content or "").strip()
    if text == "/documents":
        await _send_documents_command()
        return
    if not text:
        return

    selected_document_ids = cl.user_session.get("selected_document_ids") or []
    if not selected_document_ids:
        await cl.Message(
            content="Please select at least one document in the settings panel first."
        ).send()
        return

    # Load memory and record the user's turn. Degrade gracefully on transient
    # DB issues so the user still gets an answer.
    summary, history = "", ""
    conversation_id = cl.user_session.get("conversation_id")
    try:
        conversation_id = ai_system._ensure_conversation(conversation_id)
        cl.user_session.set("conversation_id", conversation_id)
        summary = get_conversation_summary(conversation_id)
        history = ai_system._format_history(
            get_recent_messages(conversation_id, RECENT_MESSAGES_LIMIT)
        )
        save_message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role="user",
            content=message.content,
            used_document_ids=selected_document_ids,
        )
    except Exception as e:
        print(f"chainlit on_message memory-load error: {e!r}")

    try:
        token_aiter, used_docs = await asyncio.to_thread(
            lambda: asyncio.run(
                stream_answer_with_retrieval(
                    ai_system.answer_chain,
                    message.content,
                    selected_document_ids,
                    summary=summary,
                    history=history,
                    selected_documents=", ".join(selected_document_ids),
                )
            )
        )
    except NoDocumentSelectedError as e:
        await cl.Message(content=str(e)).send()
        return
    except Exception as e:
        print(f"chainlit on_message retrieval error: {e!r}")
        await cl.Message(
            content="Sorry, something went wrong while generating the answer. Please try again."
        ).send()
        return

    msg = cl.Message(content="")
    await msg.send()
    full_answer = ""
    try:
        async for chunk in token_aiter:
            full_answer += chunk
            await msg.stream_token(chunk)
    except Exception as e:
        print(f"chainlit on_message streaming error: {e!r}")
        if not full_answer:
            full_answer = (
                "Sorry, something went wrong while generating the answer. Please try again."
            )
            await msg.stream_token(full_answer)

    msg.elements = _source_elements(used_docs)
    await msg.update()

    # Persist the assistant's turn, selection, and (periodically) the summary.
    try:
        chunk_ids = [d.metadata.get("chunk_id") for d in used_docs if d.metadata.get("chunk_id")]
        save_message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer,
            used_document_ids=selected_document_ids,
            retrieved_chunk_ids=chunk_ids,
        )
        update_conversation_selected_documents(conversation_id, selected_document_ids)
        await cl.make_async(ai_system._maybe_update_summary)(conversation_id)
    except Exception as e:
        print(f"chainlit on_message persist error: {e!r}")


if __name__ == "__main__":
    # Allow `python app.py` (or `py app.py`) in addition to `chainlit run`
    from chainlit.cli import run_chainlit

    run_chainlit(os.path.abspath(__file__))
