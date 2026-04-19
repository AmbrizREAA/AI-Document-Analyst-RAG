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
cp .env.example .env
```
Edit the .env file and add your key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
### 5. Run the application
```bash
python app.py
```
# How to Use

Run the application.
Go to the "Upload Document" tab and upload one or more PDF files.
Switch to the "Chat with Documents" tab.
Ask questions in natural language and press Enter.
The app will retrieve relevant information and provide precise answers based on the document content.

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
├── vector_stores/                     # Folder where vector databases are stored (auto-created)
└── README.md
```
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
Made Carlos by Alejandro Ambriz.
Actively seeking entry-level opportunities in Data Analyst | IT Business Analyst | AI Applications.