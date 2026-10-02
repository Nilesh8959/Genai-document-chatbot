"""
app.py
------
Streamlit front-end for the AI-Powered Document Q&A Chatbot.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    load_document,
    split_into_chunks,
    build_vector_store,
    build_qa_chain,
    answer_question,
)

load_dotenv()  # loads GROQ_API_KEY from a local .env file, if present

st.set_page_config(page_title="DocMind — Document Q&A", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------------------
# Custom styling — dark, modern "AI product" look
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        :root {
            --bg-1: #FDF6EC;
            --bg-2: #F1EBFF;
            --surface: #FFFFFF;
            --border: #EAE3F7;
            --accent: #7C5CFC;
            --accent-2: #FF7A59;
            --accent-3: #2FD9C4;
            --text: #241F36;
            --text-dim: #74708C;
        }

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 55%, #FDF1F5 100%) fixed;
            color: var(--text);
        }

        h1, h2, h3, .hero-title {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text);
        }

        /* Hero header */
        .hero-wrap {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 10px 18px;
            margin-bottom: 20px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(124, 92, 252, 0.10);
        }
        .hero-badge {
            width: 52px; height: 52px;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--accent), var(--accent-3));
            display: flex; align-items: center; justify-content: center;
            font-size: 26px;
            flex-shrink: 0;
            box-shadow: 0 8px 20px rgba(124, 92, 252, 0.35);
        }
        .hero-title {
            font-size: 27px; font-weight: 800; margin: 0;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub { font-size: 14px; color: var(--text-dim); margin: 3px 0 0 0; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFF9F0 0%, #F3EEFF 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] h2 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] label { color: var(--text) !important; font-weight: 500; }
        [data-testid="stSidebar"] .stTextInput input {
            background: var(--surface);
            color: var(--text);
            border: 1.5px solid var(--border);
            border-radius: 12px;
        }

        /* Buttons */
        .stButton button, .stButton button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent) 0%, #A78BFA 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.6em 1.3em;
            box-shadow: 0 6px 16px rgba(124, 92, 252, 0.30);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 10px 24px rgba(124, 92, 252, 0.42);
        }

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface) !important;
            border: 2px dashed #C9BBF7 !important;
            border-radius: 14px !important;
        }

        /* Status banners */
        .stAlert {
            border-radius: 14px !important;
            border: 1px solid var(--border) !important;
        }

        /* Active document chip */
        .doc-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(90deg, #EDE7FF, #FFE9E0);
            border: 1px solid var(--border);
            padding: 9px 16px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 600;
            color: var(--text);
            margin: 4px 0 18px 0;
        }
        .doc-chip .dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--accent-3);
            box-shadow: 0 0 8px var(--accent-3);
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 6px 8px;
            margin-bottom: 12px;
            box-shadow: 0 4px 14px rgba(36, 31, 54, 0.05);
        }

        /* Chat input */
        [data-testid="stChatInput"] {
            background: transparent !important;
        }
        [data-testid="stChatInput"] textarea {
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 14px !important;
        }

        /* Expander (sources) */
        [data-testid="stExpander"] {
            background: #FAF8FF;
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        /* Tips box */
        .tips-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-2);
            border-radius: 14px;
            padding: 16px 18px;
            font-size: 14px;
            color: var(--text-dim);
            margin-bottom: 20px;
            box-shadow: 0 6px 18px rgba(124, 92, 252, 0.08);
        }
        .tips-box b { color: var(--text); }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">🧠</div>
        <div>
            <p class="hero-title">DocMind</p>
            <p class="hero-sub">Ask your documents anything — in English or Hindi — with real citations.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: API key + file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    env_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Free key from https://console.groq.com/keys. "
        "Or set GROQ_API_KEY in a .env file so you don't have to paste it each time.",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload a document", type=["pdf", "txt"])

    process_clicked = st.button("🚀 Process Document", type="primary", disabled=not (api_key and uploaded_file))

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ---------------------------------------------------------------------------
# Process the uploaded document into a retrieval chain
# ---------------------------------------------------------------------------
if process_clicked and uploaded_file is not None:
    spinner_msg = "Reading document, creating embeddings, and building the vector index..."
    if "embeddings_loaded" not in st.session_state:
        spinner_msg += " (first run also downloads the embedding model — one-time, ~1-2 min)"
    with st.spinner(spinner_msg):
        suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            docs = load_document(tmp_path)
            chunks = split_into_chunks(docs)
            vector_store = build_vector_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store, api_key)
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chat_history = []
            st.session_state.embeddings_loaded = True
            st.success(f"'{uploaded_file.name}' processed into {len(chunks)} chunks. Ask away!")
        except Exception as e:
            st.error(f"Something went wrong while processing the document: {e}")
        finally:
            os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if st.session_state.qa_chain is not None:
    st.markdown(
        f"""<div class="doc-chip"><span class="dot"></span> Chatting with <b>{st.session_state.doc_name}</b></div>""",
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="tips-box">
            <b>💡 What can you ask?</b><br>
            • <b>Summaries</b> — "Summarize chapter 2 in 3 bullet points"<br>
            • <b>Specific facts</b> — "What does the document say about X?"<br>
            • <b>Definitions</b> — "How does this document define Y?"<br>
            • <b>Lists / steps</b> — "What are the steps mentioned for Z?"<br>
            • <b>Comparisons</b> — "What's the difference between A and B according to this doc?"<br>
            • <b>Follow-ups</b> — "Can you explain that point in simpler terms?"<br>
            • <b>Hindi sawal</b> — "Is document me X ke baare me kya likha hai?" (Hindi ya Hinglish me bhi pooch sakte ho)<br><br>
            Every answer shows the exact page it came from, so you can verify it yourself.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for turn in st.session_state.chat_history:
        with st.chat_message("user", avatar="🧑"):
            st.write(turn["question"])
        with st.chat_message("assistant", avatar="🧠"):
            st.write(turn["answer"])
            with st.expander("📌 Sources"):
                for s in turn["sources"]:
                    st.markdown(f"**Page {s['page']}:** {s['snippet']}")

    question = st.chat_input("Ask a question in English or Hindi...")
    if question:
        with st.chat_message("user", avatar="🧑"):
            st.write(question)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking..."):
                result = answer_question(st.session_state.qa_chain, question)
                st.write(result["answer"])
                with st.expander("📌 Sources"):
                    for s in result["sources"]:
                        st.markdown(f"**Page {s['page']}:** {s['snippet']}")
        st.session_state.chat_history.append(
            {"question": question, "answer": result["answer"], "sources": result["sources"]}
        )
else:
    st.markdown(
        """
        <div class="tips-box">
        <b>👋 Get started</b><br>
        1. Paste your free Groq API key in the sidebar<br>
        2. Upload a PDF or text file<br>
        3. Click <b>Process Document</b><br>
        4. Ask anything about it — summaries, specific facts, definitions, comparisons, and more.
        </div>
        """,
        unsafe_allow_html=True,
    )