# AI Document Analyst - RAG Application

**Intelligent PDF Document Analysis using Retrieval-Augmented Generation (RAG)**

This is my first Artificial Intelligence project. A complete application that allows users to upload PDF documents and ask questions in natural language, receiving accurate and context-aware answers.

Built with limited hardware (GTX 1060 + 32 GB RAM), demonstrating efficient resource optimization and practical AI development.

---

## Features

- **PDF Upload & Processing**: Intelligent chunking and persistent vector storage with FAISS
- **Semantic Search**: Accurate retrieval using embeddings
- **Intelligent Responses**: Powered by Groq + Llama 3.1 with advanced prompt engineering to reduce hallucinations
- **User-Friendly Interface**: Clean and interactive UI built with Gradio
- **Document Persistence**: Each uploaded document gets its own FAISS vector database
- **API Key Security**: Environment variables management via `.env`

---

## Tech Stack

- **Python 3.10**
- **LangChain** (with langchain-classic for compatibility)
- **Groq** + Llama 3.1 (fast and cost-effective inference)
- **Hugging Face Embeddings** (`all-MiniLM-L6-v2`)
- **FAISS-cpu**
- **Gradio**
- **Sentence-Transformers**

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/AmbrizREAA/AI-RAG-Application.git
cd AI-RAG-Application

### 2. Create virtual environment (recommended)
'''bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate


