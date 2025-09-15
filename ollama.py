
# app_ollama_authenticated_fixed.py
# ------------------------------------------
# Streamlit + Ollama RAG demo with authentication and SQLite storage
# CORRECTED VERSION for AcademiaX
# ------------------------------------------
import os
import sqlite3
import hashlib
import re
from io import BytesIO
from typing import List, Tuple, Optional
import json
import requests
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Database setup
DB_PATH = "academiax.db"

def create_tables():
    """Create main authentication tables with user roles"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'student' CHECK (role IN ('student', 'teacher')),
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prompt TEXT,
                response TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                content BLOB,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
        conn.commit()

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                      (username, hash_pw(password)))
            conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def authenticate_user(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username=? AND password_hash=?",
                  (username, hash_pw(password)))
        result = c.fetchone()
        if result:
            return True, result[0]
        return False, None

def save_chat(user_id, prompt, response):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_history (user_id, prompt, response) VALUES (?, ?, ?)",
            (user_id, prompt, response))
        conn.commit()

def get_chat_history(user_id, limit=20):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT date, prompt, response FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        )
        return c.fetchall()

def save_upload(user_id, filename, content):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO uploads (user_id, filename, content) VALUES (?, ?, ?)",
            (user_id, filename, content.read()))
        conn.commit()

def get_uploads(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT filename, content FROM uploads WHERE user_id=? ORDER BY upload_date DESC",
            (user_id,))
        return c.fetchall()

def clear_user_data(user_id):
    """Clear all user data (uploads, chat history)"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM uploads WHERE user_id=?", (user_id,))
        conn.commit()

# --- PDF TEXT CHUNKING ---
def pdfs_to_text_chunks(files: List[BytesIO], chunk_size: int = 900, overlap: int = 150) -> List[str]:
    texts = []
    for f in files:
        try:
            f.seek(0)  # Reset file pointer
            reader = PdfReader(f)
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                t = re.sub(r"[ \t]+", " ", t)
                pages.append(t.strip())
            doc = "\n".join(pages).strip()
            if doc:
                texts.append(doc)
        except Exception as e:
            st.warning(f"Error reading PDF: {str(e)}")
            continue

    if not texts:
        return []

    full_text = "\n\n".join(texts)
    # FIXED: Simple sentence split - correct regex syntax
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= chunk_size:
            buf += (" " if buf else "") + s
        else:
            if buf:
                chunks.append(buf.strip())
            tail = buf[-overlap:] if buf else ""
            buf = (tail + " " + s).strip()
    if buf:
        chunks.append(buf.strip())
    return [c.strip() for c in chunks if c and len(c) > 20]

# --- TF-IDF VECTOR STORE ---
def build_vector_store(chunks: List[str]):
    if not chunks:
        return None, None
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=1, stop_words='english')
        matrix = vectorizer.fit_transform(chunks)
        return vectorizer, matrix
    except Exception as e:
        st.error(f"Error building vector store: {str(e)}")
        return None, None

def retrieve(query: str, vectorizer, matrix, chunks: List[str], k: int = 5) -> List[Tuple[str, float]]:
    if not vectorizer or matrix is None or not query:
        return []
    try:
        qv = vectorizer.transform([query])
        sims = cosine_similarity(qv, matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        return [(chunks[i], float(sims[i])) for i in idxs if sims[i] > 0.01]
    except Exception as e:
        st.error(f"Error during retrieval: {str(e)}")
        return []

# --- OLLAMA LOCAL LLM CHAT ---
def test_ollama_connection(base_url: str) -> Tuple[bool, str]:
    """Test if Ollama server is running and accessible."""
    try:
        url = base_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        if models:
            model_names = [m.get("name", "unknown") for m in models]
            return True, f"Connected! Available models: {', '.join(model_names)}"
        else:
            return True, "Connected! No models found. Run 'ollama pull llama3.2' to download a model."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama server. Make sure Ollama is running."
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def call_local_llm_chat(
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    model: str,
    temperature: float = 0.2,
    timeout: int = 120
) -> str:
    try:
        url = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        content = msg.get("content", "")
        return content or "No response from LLM."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama server. Please check if Ollama is running."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Model '{model}' not found. Try pulling it with 'ollama pull {model}'"
        return f"HTTP error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Local LLM error: {e}"

# --- RAG ANSWER GENERATION ---
def rag_answer(query: str, chunks_scored: List[Tuple[str, float]], llm_url: str, llm_model: str, temp: float) -> str:
    if not chunks_scored:
        return "I couldn't find relevant information in the uploaded PDFs to answer your question."

    # Prepare context from retrieved chunks
    context_parts = []
    total_length = 0
    max_context_length = 8000

    for chunk, score in chunks_scored:
        if total_length + len(chunk) <= max_context_length:
            context_parts.append(f"- {chunk}")
            total_length += len(chunk)
        else:
            break

    context = "\n\n".join(context_parts)
    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the provided context. "
        "If the answer cannot be found in the context, say that you don't have enough information. "
        "Be concise and accurate.\n\n"
        f"Context:\n{context}"
    )
    user_prompt = f"Question: {query}\n\nPlease provide a clear and concise answer based on the context above."

    answer = call_local_llm_chat(system_prompt, user_prompt, llm_url, llm_model, temperature=temp)
    return answer

# --- STREAMLIT UI ---
def main():
    create_tables()

    st.set_page_config(
        page_title="🎓 AcademiaX Student Portal", 
        page_icon="🎓", 
        layout="wide"
    )

    # Initialize session state
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
        st.session_state["username"] = None
        st.session_state["student_chunks"] = []
        st.session_state["student_vec"] = None
        st.session_state["student_mat"] = None

    st.title("🎓 AcademiaX Student Portal")

    # --- AUTHENTICATION SECTION ---
    if st.session_state["user_id"] is None:
        st.subheader("🔑 Student Login")

        with st.form("student_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button and username and password:
                success, user_id = authenticate_user(username, password)
                if success:
                    st.session_state["user_id"] = user_id
                    st.session_state["username"] = username
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        st.info("💡 Create your account in the main AcademiaX platform first.")
        st.stop()

    # --- LOGGED IN STUDENT INTERFACE ---
    st.sidebar.success(f"👨‍🎓 Student: **{st.session_state['username']}**")

    if st.sidebar.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # --- LLM CONFIGURATION ---
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ LLM Settings")
    llm_base_url = st.sidebar.text_input("Ollama Base URL", value="http://localhost:11434")
    llm_model = st.sidebar.text_input("Model", value="llama3.2")
    llm_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    # Test connection
    if st.sidebar.button("🔍 Test Connection"):
        with st.spinner("Testing..."):
            is_connected, message = test_ollama_connection(llm_base_url)
            if is_connected:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)

    # --- MAIN APPLICATION ---
    tab1, tab2 = st.tabs(["📚 Knowledge Base & Chat", "📊 Dashboard"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📚 Document Management")

            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF files", 
                accept_multiple_files=True, 
                type=["pdf"],
                help="Upload PDFs to build your personal knowledge base"
            )

            if uploaded_files:
                for file in uploaded_files:
                    file.seek(0)
                    save_upload(st.session_state["user_id"], file.name, file)
                st.success(f"Uploaded {len(uploaded_files)} files!")

            # Show existing uploads
            user_uploads = get_uploads(st.session_state["user_id"])
            if user_uploads:
                st.write(f"**Your uploaded files ({len(user_uploads)} total):**")
                for i, (filename, _) in enumerate(user_uploads[:5]):
                    st.write(f"• {filename}")
                if len(user_uploads) > 5:
                    st.write(f"... and {len(user_uploads) - 5} more")

            # Build knowledge base
            if st.button("🔨 Build Knowledge Base", type="primary"):
                if user_uploads:
                    with st.spinner("Processing your documents..."):
                        kb_files = [BytesIO(content) for _, content in user_uploads]
                        chunks = pdfs_to_text_chunks(kb_files, chunk_size=900, overlap=150)

                        if chunks:
                            vec, mat = build_vector_store(chunks)
                            st.session_state["student_chunks"] = chunks
                            st.session_state["student_vec"] = vec
                            st.session_state["student_mat"] = mat
                            st.success(f"✅ Built knowledge base with {len(chunks)} chunks!")
                        else:
                            st.error("❌ Could not extract text from your PDFs.")
                else:
                    st.warning("⚠️ Please upload some PDF files first.")

            # KB Status
            kb_size = len(st.session_state.get("student_chunks", []))
            st.metric("Knowledge Base Size", f"{kb_size} chunks")

        with col2:
            st.subheader("💬 Chat Interface")

            # Chat input
            with st.form("chat_form", clear_on_submit=True):
                chat_input = st.text_area("Ask a question about your documents:", height=100)
                ask_button = st.form_submit_button("Ask", type="primary")

            if ask_button and chat_input and st.session_state.get("student_vec"):
                with st.spinner("Thinking..."):
                    # Retrieve relevant chunks
                    chunks_scored = retrieve(
                        chat_input, 
                        st.session_state["student_vec"], 
                        st.session_state["student_mat"], 
                        st.session_state["student_chunks"], 
                        k=6
                    )

                    # Generate answer
                    answer = rag_answer(chat_input, chunks_scored, llm_base_url, llm_model, llm_temp)

                    # Save to database
                    save_chat(st.session_state["user_id"], chat_input, answer)

                    # Display answer
                    st.write("**Answer:**")
                    st.write(answer)

                    # Show sources
                    if chunks_scored:
                        with st.expander("📖 Sources Used"):
                            for i, (chunk, score) in enumerate(chunks_scored, 1):
                                st.write(f"**Source {i}** (relevance: {score:.3f})")
                                st.write(f"_{chunk[:200]}..._")
                                st.divider()

            elif ask_button and chat_input:
                st.warning("⚠️ Please build your knowledge base first by uploading PDFs.")

    with tab2:
        st.subheader("📊 Your Learning Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            # Chat history
            st.write("**Recent Chat History**")
            chat_history = get_chat_history(st.session_state["user_id"], limit=10)

            if chat_history:
                for date, prompt, response in chat_history:
                    with st.expander(f"[{date}] {prompt[:50]}..."):
                        st.write(f"**Q:** {prompt}")
                        st.write(f"**A:** {response}")
            else:
                st.info("No chat history yet. Start asking questions!")

        with col2:
            # Statistics
            st.write("**Your Statistics**")
            total_chats = len(get_chat_history(st.session_state["user_id"], limit=1000))
            total_uploads = len(get_uploads(st.session_state["user_id"]))

            st.metric("Total Questions Asked", total_chats)
            st.metric("Documents Uploaded", total_uploads)
            st.metric("Knowledge Base Chunks", kb_size)

if __name__ == "__main__":
    main()

