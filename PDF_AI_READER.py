import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
    
load_dotenv()

class DocumentReaderAI:
    """
    Core class handling PDF ingestion and RAG-based response generation with local vector persistence.
    """
    def __init__(self):
        print("Loading model...")
        
        # 1. Model used to convert text to numbers (Embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
                
        mi_api_key = os.getenv("GROQ_API_KEY") 
        
        self.llm = ChatGroq(
            temperature=0,             # 0 means no creativity or hallucinations
            groq_api_key=mi_api_key,
            model_name="llama-3.1-8b-instant" # Llama 3 by meta, optimized by groq
        )
        
        self.vector_store = None

    def process_document(self, file_path: str) -> str:
        """Reads the PDF, divides it and creates the vectorial base. The PDF is saved locally."""
        if not file_path:
            return "Please enter a PDF document."
        
        try:
            
            base_name = os.path.basename(file_path).replace(".pdf", "")
                    
            import re
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', base_name)
                        
            index_folder = os.path.join(os.getcwd(), "vector_stores", f"{safe_name}_faiss")
            
            # --- Logic of storage ---
            if os.path.exists(index_folder):
                print(f"Loading memory from: {index_folder}")
                self.vector_store = FAISS.load_local(
                    index_folder, 
                    self.embeddings,
                    allow_dangerous_deserialization=True 
                )
                return "The PDF was already in the storage, please ask a question"

            # --- New process ---
            print("Loading new document.")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
            
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                        
            os.makedirs(index_folder, exist_ok=True)
            self.vector_store.save_local(index_folder)
            
            return f"PDF document processed sucessfully, please ask a question"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def answer_question(self, question: str) -> str:
        """Searches document using FAISS and writes the answer using Qwen."""
        if not self.vector_store:
            return "You need to upload and process a PDF first."
        if not question:
            return "Please ask a question."

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

        Question: {question}

        Direct Answer:
        """
       
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}  
        )
        
        respuesta = qa_chain.invoke(question)
        return respuesta['result']


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
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="1. Load PDF", file_types=[".pdf"])
                process_btn = gr.Button("Process Document", variant="primary")
                status_output = gr.Textbox(label="System Status", interactive=False)
            
            with gr.Column(scale=2):
                question_input = gr.Textbox(label="2. Ask something about the text", lines=2)
                answer_btn = gr.Button("Submit Question")
                answer_output = gr.Textbox(label="AI Response", lines=5)

        
        process_btn.click(fn=ai_system.process_document, inputs=pdf_input, outputs=status_output)
        answer_btn.click(fn=ai_system.answer_question, inputs=question_input, outputs=answer_output)

    return interfaz

if __name__ == "__main__":
    app = create_interface()
    app.launch()