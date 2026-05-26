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

VECTOR_STORES_DIR = os.path.join(os.getcwd(), "vector_stores")


class DocumentReaderAI:
    """
    Core class handling PDF ingestion and RAG-based response generation with local vector persistence.
    Stateless w.r.t. user data: per-session vector stores live in gr.State, not on this instance.
    """
    def __init__(self):
        print("Loading model...")

        # 1. Model used to convert text to numbers (Embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        mi_api_key = os.getenv("GROQ_API_KEY")

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
        # Modern LangChain v0.2/v0.3 stuff-documents chain (replaces RetrievalQA)
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)

    def process_document(self, file_path: str, vector_store):
        """Reads the PDF, divides it and creates the vectorial base. The PDF is saved locally.
        Returns (status, vector_store) so Gradio holds the store in per-session gr.State."""
        if not file_path:
            return "Please enter a PDF document.", vector_store

        try:
            base_name = os.path.basename(file_path).replace(".pdf", "")
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', base_name)
            index_folder = os.path.join(VECTOR_STORES_DIR, f"{safe_name}_faiss")

            # --- Logic of storage ---
            if os.path.exists(index_folder):
                print(f"Loading memory from: {index_folder}")
                vector_store = FAISS.load_local(
                    index_folder,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return "The PDF was already in the storage, please ask a question", vector_store

            # --- New process ---
            print("Loading new document.")
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)

            vector_store = FAISS.from_documents(chunks, self.embeddings)

            os.makedirs(index_folder, exist_ok=True)
            vector_store.save_local(index_folder)

            return "PDF document processed sucessfully, please ask a question", vector_store

        except Exception as e:
            return f"Error: {str(e)}", vector_store

    def load_existing(self, store_name: str):
        """Loads a previously processed FAISS store from the vector_stores directory."""
        if not store_name:
            return "Please select a database.", None
        try:
            index_folder = os.path.join(VECTOR_STORES_DIR, store_name)
            print(f"Loading memory from: {index_folder}")
            vs = FAISS.load_local(
                index_folder,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return f"Loaded '{store_name}'. Please ask a question.", vs
        except Exception as e:
            return f"Error: {str(e)}", None

    def answer_question(self, question: str, vector_store) -> str:
        """Searches document using FAISS and writes the answer using Llama."""
        if vector_store is None:
            return "You need to upload and process a PDF first."
        if not question:
            return "Please ask a question."

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        rag_chain = create_retrieval_chain(retriever, self.combine_docs_chain)

        respuesta = rag_chain.invoke({"input": question})
        return respuesta["answer"]


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
    app = create_interface()
    app.launch()
