"""
rag_engine.py
--------------
Core Retrieval-Augmented Generation (RAG) logic for the Document Q&A Chatbot.

Pipeline:
1. Load a document (PDF or TXT)
2. Split it into overlapping chunks
3. Convert chunks into embeddings (free, local HuggingFace model - no API key needed)
4. Store embeddings in a FAISS vector index
5. On a user question: embed the question, retrieve the most relevant chunks,
   and pass them + the question to an LLM (via Groq's free API) to generate
   a grounded answer with source citations.
"""

import os
from typing import List

import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate

# Use all available CPU cores for embedding computation (big speed win on
# multi-page documents, since embedding is the slowest step).
torch.set_num_threads(os.cpu_count() or 4)

# Cache the embedding model as a module-level singleton so it is loaded
# from disk/memory only ONCE per app session, not every time a document
# is processed (re-loading the model was a big chunk of the "wait time").
_EMBEDDINGS_CACHE = {}


def _get_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    if model_name not in _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": 64,       # embed many chunks at once instead of one-by-one
                "normalize_embeddings": True,
            },
        )
    return _EMBEDDINGS_CACHE[model_name]


# Custom prompt: keeps answers grounded in the retrieved context, and
# explicitly tells the model to reply in the same language the question
# was asked in (English, Hindi/Devanagari, or Hinglish).
QA_PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant answering questions about a document.\n"
        "Use ONLY the context below to answer. If the answer isn't in the "
        "context, say you don't know based on the document.\n"
        "Always reply in the SAME language the question was asked in "
        "(English, Hindi/Devanagari, or Hinglish) — do not switch languages.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)


# ---------------------------------------------------------------------------
# 1. Load document
# ---------------------------------------------------------------------------
def load_document(file_path: str) -> List[Document]:
    """Load a PDF or TXT file into a list of LangChain Document objects."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


# ---------------------------------------------------------------------------
# 2. Split into chunks
# ---------------------------------------------------------------------------
def split_into_chunks(
    documents: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split long documents into smaller overlapping chunks for better retrieval.

    chunk_size was increased from 800 -> 1200 characters: this means FEWER
    total chunks for the same document, which directly means fewer
    embeddings to compute. This is the single biggest lever for speed on
    large (50+ page) documents, with only a small, acceptable trade-off in
    retrieval granularity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 3 & 4. Embed chunks and build a FAISS vector store
# ---------------------------------------------------------------------------
def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed chunks locally (free, no API key) and store them in FAISS.

    Uses a multilingual embedding model (50+ languages, including Hindi),
    a cached model instance (loaded once, reused after), and batched
    encoding (64 chunks at a time, all CPU cores) for speed on large docs.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ---------------------------------------------------------------------------
# 5. Build the RAG question-answering chain
# ---------------------------------------------------------------------------
def build_qa_chain(vector_store: FAISS, groq_api_key: str, model_name: str = "openai/gpt-oss-20b"):
    """
    Create a RetrievalQA chain:
    - Retriever: top-4 most similar chunks from FAISS
    - LLM: Groq-hosted open model (free tier, no cost)
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0.2,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return qa_chain


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------
def answer_question(qa_chain, question: str) -> dict:
    """Run a question through the QA chain and return the answer + sources."""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])

    source_info = []
    for doc in sources:
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:150].replace("\n", " ") + "..."
        source_info.append({"page": page, "snippet": snippet})

    return {"answer": answer, "sources": source_info}


if __name__ == "__main__":
    # Simple command-line smoke test (requires a sample file and GROQ_API_KEY env var)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY environment variable to run this test.")
        raise SystemExit(1)

    sample_path = "sample_docs/sample.txt"
    docs = load_document(sample_path)
    chunks = split_into_chunks(docs)
    store = build_vector_store(chunks)
    chain = build_qa_chain(store, api_key)

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        result = answer_question(chain, q)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - Page {s['page']}: {s['snippet']}")