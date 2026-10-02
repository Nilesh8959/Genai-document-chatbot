# 📄 AI-Powered Document Q&A Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF or
text document and ask natural-language questions about it. Answers are
grounded in the document's actual content, with page-level source citations
for every response — reducing hallucination compared to a plain LLM chatbot.

Built to understand, end-to-end, how modern GenAI applications are built:
document ingestion → chunking → embeddings → vector search → LLM generation.

## ✨ Features

- 📁 Upload any PDF or `.txt` file
- 🌐 Works with **English and Hindi (Devanagari) documents**, including mixed-language content — ask questions in English, Hindi, or Hinglish
- ✂️ Automatic chunking with overlap for better context retrieval
- 🧠 Free, local **multilingual embeddings** via HuggingFace `sentence-transformers` (no API cost)
- 🔍 Semantic similarity search using a **FAISS** vector store
- 💬 Grounded answers generated via **Groq's** free, fast LLM API (Llama 3.1)
- 📌 Source citations (page number + snippet) shown for every answer
- 🖥️ Simple, clean **Streamlit** chat interface

## 🏗️ Architecture

```
 User Document (PDF/TXT)
        │
        ▼
 ┌─────────────────┐
 │  Document Loader │  (PyPDFLoader / TextLoader)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │   Text Splitter  │  (chunk_size=800, overlap=120)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │    Embeddings    │  (sentence-transformers/all-MiniLM-L6-v2)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │   FAISS Index    │  (vector store)
 └─────────────────┘
        │
   user question ──► embed ──► similarity search (top-4 chunks)
        │
        ▼
 ┌─────────────────┐
 │   Groq LLM       │  (retrieved chunks + question → grounded answer)
 └─────────────────┘
        │
        ▼
   Answer + Source Citations
```

## 🛠️ Tech Stack

| Component        | Technology                                      |
|-------------------|--------------------------------------------------|
| Orchestration     | LangChain                                       |
| Vector Store      | FAISS                                           |
| Embeddings        | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (local, free, 50+ languages incl. Hindi) |
| LLM               | Groq API — `llama-3.1-8b-instant` (free tier)   |
| UI                | Streamlit                                       |
| Document Parsing  | pypdf                                           |

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/genai-document-chatbot.git
cd genai-document-chatbot
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Get a free Groq API key
Sign up at [console.groq.com/keys](https://console.groq.com/keys) (free, no credit card).

Copy `.env.example` to `.env` and paste your key:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`), upload a
document from `sample_docs/` (or your own PDF), and start asking questions.

## 📂 Project Structure
```
genai-document-chatbot/
├── app.py              # Streamlit UI
├── rag_engine.py        # Core RAG pipeline (loading, chunking, embeddings, retrieval, QA chain)
├── sample_docs/
│   └── sample.txt        # Sample document to test with
├── requirements.txt
├── .env.example
└── README.md
```

## 🔮 Future Improvements
- Support multiple documents in one session
- Add conversation memory for follow-up questions
- Swap FAISS for a hosted vector DB (Pinecone/ChromaDB) for persistence
- Add evaluation metrics (answer relevancy, faithfulness) using RAGAS

## 📝 License
MIT