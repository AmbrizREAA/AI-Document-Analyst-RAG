# AI Document Analyst - RAG Application

**Intelligent PDF Document Analysis using Retrieval-Augmented Generation (RAG)**

This is my first Artificial Intelligence project. A complete application that allows users to upload PDF documents and ask questions in natural language, receiving accurate and context-aware answers.

Built with limited hardware (GTX 1060 + 32 GB RAM), demonstrating efficient resource optimization and practical AI development.

---

## Features

- **PDF Upload & Processing**: Intelligent chunking and persistent vector storage with FAISS
- **One PDF at a time**: The current UI works with a single active PDF per session (you can switch between previously processed PDFs via the dropdown)
- **Semantic Search**: Accurate retrieval using embeddings
- **Intelligent Responses**: Powered by Groq + Llama 3.1 with advanced prompt engineering to reduce hallucinations
- **User-Friendly Interface**: Clean and interactive UI built with Gradio
- **Document Persistence**: Each uploaded document gets its own FAISS vector database
- **Existing Database Dropdown**: Already-processed PDFs are auto-detected at startup and can be loaded instantly without re-upload
- **Per-Session State**: Each user gets an isolated `gr.State()` vector store — no cross-talk between concurrent sessions
- **Modern Retrieval Chain**: Uses the current LangChain `create_retrieval_chain` + `create_stuff_documents_chain` API (replaces the deprecated `RetrievalQA`)
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
git clone https://github.com/AmbrizREAA/AI-Document-Analyst-RAG.git
cd AI-Document-Analyst-RAG
```
### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Linux / macOS
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```
Edit the `.env` file and add your key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
The app validates `GROQ_API_KEY` at startup and exits with a clear message if it is missing or still set to the placeholder.

### 5. Run the application
```bash
python app.py
```
# How to Use

Run the application. The UI offers two entry points:

1. **Load an existing database (optional)** — the dropdown lists every FAISS index previously saved under `vector_stores/`. Pick one to start asking questions immediately.
2. **Upload a new PDF** — choose a file, click *Process Document*. If the same PDF has been processed before, its stored index is reused; otherwise the app chunks, embeds, and saves a new one.

Then type a question in natural language and press *Submit Question*. The app retrieves the top-k relevant chunks and produces a precise, context-bound answer.

Example questions:

"What are the main findings of the document?"
"What does the report recommend about topic X?"
"Summarize the key points on page 5."

## Project Structure
```text
AI-Document-Analyst-RAG/
├── app.py                             # Main application file
├── requirements.txt
├── .env.example
├── .gitignore
├── vector_stores/                     # Folder where FAISS indexes are stored (auto-created, git-ignored)
└── README.md
```
---
## Security Notes

- **FAISS deserialization**: This app loads FAISS indexes with `allow_dangerous_deserialization=True`, which uses Python `pickle` under the hood. Loading a pickle file from an untrusted source can execute arbitrary code on your machine. The app only loads indexes from the local `vector_stores/` directory (paths are validated to prevent traversal), and you should **never** copy a `vector_stores/<name>_faiss/` folder from someone you do not trust into this project.
- **API key handling**: `GROQ_API_KEY` is loaded from `.env`, which is git-ignored. Do not paste your key into source files, commit messages, or screenshots.
- **Generated vector stores are not for git**: The `vector_stores/` folder contains embeddings derived from your PDFs and is excluded by `.gitignore`. If older commits accidentally tracked any `*_faiss/` folder, untrack it with `git rm -r --cached vector_stores/<that_folder>` and commit the removal.

---
### Future Improvements

- Support for additional file formats (Word, Excel, plain text)
- Persistent chat history
- Deployment to Hugging Face Spaces or Streamlit Sharing
- Automated response evaluation (RAGAS)

### License 
This project is open source under the MIT License. Feel free to use, modify, and learn from it.

#### Thank you for visiting!

If you like the project, please give it a ⭐. Any feedback or suggestions are welcome.
Made by Carlos Alejandro Ambriz.
Actively seeking entry-level opportunities in Data Analyst | IT Business Analyst | AI Applications.