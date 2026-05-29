import gradio as gr
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORES_DIR = os.path.realpath(os.path.join(os.getcwd(), "vector_stores"))


def _safe_index_path(store_name: str) -> str | None:
    """Resolve a vector-store folder under VECTOR_STORES_DIR.

    Returns the absolute path only if the resolved location stays inside
    VECTOR_STORES_DIR. Returns None on attempted path traversal (e.g. "..", "/etc/...").
    """
    if not store_name or not isinstance(store_name, str):
        return None
    # Block path separators and parent refs outright — store names are flat folder names.
    if any(sep in store_name for sep in ("/", "\\")) or store_name in (".", ".."):
        return None
    candidate = os.path.realpath(os.path.join(VECTOR_STORES_DIR, store_name))
    root = VECTOR_STORES_DIR
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    return candidate


class DocumentReaderAI:
    """
    Core class handling PDF ingestion and RAG-based response generation with local vector persistence.
    Stateless w.r.t. user data: per-session vector stores live in gr.State, not on this instance.
    """
    def __init__(self):
        print("Loading model...")

        mi_api_key = os.getenv("GROQ_API_KEY")
        if not mi_api_key or mi_api_key.strip() in ("", "your_groq_api_key_here"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://console.groq.com/keys before starting the app."
            )

        # 1. Model used to convert text to numbers (Embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.llm = ChatGroq(
            temperature=0,                    # 0 means no creativity or hallucinations
            groq_api_key=mi_api_key,
            model_name="llama-3.1-8b-instant" # Llama 3 by meta, optimized by groq
        )

        prompt_template = """
        You are a highly precise information extraction tool. Your ONLY job is to extract the exact answer to the user's question from the provided context.

        STRICT RULES:
        1. EXTRACT AND STOP: Provide ONLY the direct answer to the question. You are FORBIDDEN from summarizing the rest of the context.
        2. NO EXTRA FLUFF: Do not add conclusions, greetings, or extra sections (like examples or SQL commands) unless the user explicitly asked for them.
        3. CONCISENESS: Keep the answer as short as possible. Use bullet points if you need to list items.
        4. LANGUAGE: Answer strictly in the same language the user used to ask the question.
        5. UNKNOWN: If the direct answer is not in the context, output exactly: "I don't have enough information / No tengo suficiente información".
        6. CRITICAL: DO NOT generate follow-up questions, user prompts, or conversational filler after providing the direct answer. Stop generating text immediately after your explanation.

        Context:
        {context}

        Question: {input}

        Direct Answer:
        """

        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # Modern LangChain stuff-documents chain. RetrievalQA is deprecated upstream;
        # this app already uses the recommended replacement (create_retrieval_chain +
        # create_stuff_documents_chain), so no further migration is needed.
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)

    def _load_faiss(self, index_folder: str):
        """Load a FAISS index from disk.

        SECURITY: FAISS persists its docstore via Python pickle. Deserializing a pickle
        file from an untrusted source can execute arbitrary code. We only load folders
        we created ourselves under VECTOR_STORES_DIR, callers must validate the path
        with _safe_index_path() first, and users should never load indexes obtained
        from third parties into this app.
        """
        print(f"Loading memory from: {index_folder}")
        return FAISS.load_local(
            index_folder,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def process_document(self, file_path: str, vector_store):
        """Reads the PDF, splits it into chunks and builds a FAISS vector index.

        The FAISS index (not the original PDF) is persisted locally so the same PDF
        can be reused on later runs. Returns (status, vector_store) so Gradio holds
        the store in per-session gr.State.
        """
        if not file_path:
            return "Please upload a PDF document.", vector_store

        try:
            base_name = os.path.basename(file_path)
            if base_name.lower().endswith(".pdf"):
                base_name = base_name[:-4]
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', base_name).strip("_")
            if not safe_name:
                return "Invalid PDF filename. Please rename the file and try again.", vector_store

            folder_name = f"{safe_name}_faiss"
            index_folder = _safe_index_path(folder_name)
            if index_folder is None:
                return "Invalid storage path resolved for this PDF.", vector_store

            # --- Reuse existing index if we built one for this PDF before ---
            if os.path.isdir(index_folder):
                vector_store = self._load_faiss(index_folder)
                return "The PDF was already in storage. Please ask a question.", vector_store

            # --- New document: chunk, embed, persist ---
            print("Loading new document.")
            if not os.path.isfile(file_path):
                return "The uploaded file could not be found on disk.", vector_store

            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not documents:
                return "The PDF appears to be empty or unreadable.", vector_store

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                return "No extractable text was found in the PDF.", vector_store

            vector_store = FAISS.from_documents(chunks, self.embeddings)

            os.makedirs(index_folder, exist_ok=True)
            vector_store.save_local(index_folder)

            return "PDF document processed successfully. Please ask a question.", vector_store

        except FileNotFoundError:
            return "The uploaded file could not be found on disk.", vector_store
        except PermissionError:
            return "Permission denied while reading the PDF or writing the index.", vector_store
        except Exception as e:
            print(f"process_document error: {e!r}")
            return "An unexpected error occurred while processing the PDF. Please try a different file.", vector_store

    def load_existing(self, store_name: str):
        """Loads a previously processed FAISS store from the vector_stores directory."""
        if not store_name:
            return "Please select a database.", None
        try:
            index_folder = _safe_index_path(store_name)
            if index_folder is None or not os.path.isdir(index_folder):
                return "Database not found.", None
            vs = self._load_faiss(index_folder)
            return f"Loaded '{store_name}'. Please ask a question.", vs
        except Exception as e:
            print(f"load_existing error: {e!r}")
            return "Could not load the selected database.", None

    def answer_question(self, question: str, vector_store) -> str:
        """Retrieves the top-k relevant chunks from FAISS and produces an answer via the LLM."""
        if vector_store is None:
            return "You need to upload and process a PDF first."
        if not question or not question.strip():
            return "Please ask a question."

        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            rag_chain = create_retrieval_chain(retriever, self.combine_docs_chain)
            respuesta = rag_chain.invoke({"input": question})
            return respuesta["answer"]
        except Exception as e:
            print(f"answer_question error: {e!r}")
            return "Sorry, something went wrong while generating the answer. Please try again."


def list_existing_stores():
    """Scans the vector_stores directory on startup for already-processed databases."""
    if not os.path.isdir(VECTOR_STORES_DIR):
        return []
    return sorted(
        d for d in os.listdir(VECTOR_STORES_DIR)
        if os.path.isdir(os.path.join(VECTOR_STORES_DIR, d)) and d.endswith("_faiss")
    )


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
        gr.Markdown("Submit a PDF and ask a question about its content. Powered by Llama 3, LangChain, FAISS, and Groq.")

        gr.HTML("<hr>")

        # Per-session vector store (fixes shared-state cross-talk between users)
        vs_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                existing_dropdown = gr.Dropdown(
                    choices=list_existing_stores(),
                    label="0. Load an existing database (optional)",
                    interactive=True
                )
                pdf_input = gr.File(label="1. Load PDF", file_types=[".pdf"])
                process_btn = gr.Button("Process Document", variant="primary")
                status_output = gr.Textbox(label="System Status", interactive=False)

            with gr.Column(scale=2):
                question_input = gr.Textbox(label="2. Ask something about the text", lines=2)
                answer_btn = gr.Button("Submit Question")
                answer_output = gr.Textbox(label="AI Response", lines=5)


        process_btn.click(
            fn=ai_system.process_document,
            inputs=[pdf_input, vs_state],
            outputs=[status_output, vs_state]
        )
        existing_dropdown.change(
            fn=ai_system.load_existing,
            inputs=existing_dropdown,
            outputs=[status_output, vs_state]
        )
        answer_btn.click(
            fn=ai_system.answer_question,
            inputs=[question_input, vs_state],
            outputs=answer_output
        )

    return interfaz

if __name__ == "__main__":
    try:
        app = create_interface()
    except RuntimeError as e:
        # Surface configuration errors (e.g. missing GROQ_API_KEY) cleanly at startup.
        print(f"\n[CONFIG ERROR] {e}\n")
        raise SystemExit(1)
    app.launch()
