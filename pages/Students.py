
# pages/Student.py
# --------------------------------------------------
# Student workspace: upload PDFs, build KB, chat over PDFs
# FIXED VERSION for AcademiaX
# --------------------------------------------------

import io
import streamlit as st
from app_ollama_authenticated_fixed import (
    get_uploads, save_upload, pdfs_to_text_chunks,
    build_vector_store, retrieve, rag_answer,
    save_chat, get_chat_history, test_ollama_connection
)

# Session gate
if "user_id" not in st.session_state or st.session_state["user_id"] is None:
    st.error("Please log in from the AcademiaX home page.")
    st.stop()

st.set_page_config(page_title="AcademiaX — Student", page_icon="🎓", layout="wide")
st.title("🎓 Student — Study & Chat with Your PDFs")

# LLM config
st.sidebar.subheader("⚙️ Local LLM (Ollama)")
llm_base_url = st.sidebar.text_input("Base URL", value="http://localhost:11434")
llm_model = st.sidebar.text_input("Model", value="llama3.2")
llm_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

if st.sidebar.button("Test Ollama"):
    ok, msg = test_ollama_connection(llm_base_url)
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

# Init KB state
for key in ["student_chunks", "student_vec", "student_mat"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "chunks" in key else None

colL, colR = st.columns([1, 1])

with colL:
    st.subheader("📚 Upload PDFs")

    files = st.file_uploader(
        "Upload one or more PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if files:
        for f in files:
            f.seek(0)
            save_upload(st.session_state["user_id"], f.name, f)
        st.success(f"Uploaded {len(files)} file(s).")

    st.write("**Your uploads:**")
    uploads = get_uploads(st.session_state["user_id"])

    if not uploads:
        st.info("No files yet. Upload PDFs to get started.")
    else:
        for i, (name, _) in enumerate(uploads[:8]):
            st.write(f"• {name}")
        if len(uploads) > 8:
            st.caption(f"+ {len(uploads) - 8} more…")

    if st.button("🔨 Build Knowledge Base", type="primary"):
        if not uploads:
            st.warning("Upload PDFs first.")
        else:
            with st.spinner("Processing your documents..."):
                kb_files = [io.BytesIO(content) for (name, content) in uploads]
                chunks = pdfs_to_text_chunks(kb_files, chunk_size=900, overlap=150)

                if not chunks:
                    st.error("Couldn't extract text (are these scanned PDFs?).")
                else:
                    vec, mat = build_vector_store(chunks)
                    st.session_state["student_chunks"] = chunks
                    st.session_state["student_vec"] = vec
                    st.session_state["student_mat"] = mat
                    st.success(f"KB built with {len(chunks)} chunks.")

    kb_size = len(st.session_state.get("student_chunks") or [])
    st.metric("KB chunks", kb_size)

with colR:
    st.subheader("💬 Ask your documents")

    with st.form("chat_form", clear_on_submit=True):
        q = st.text_area(
            "Your question", 
            height=100, 
            placeholder="Ask anything grounded in your uploaded PDFs…"
        )
        ask_button = st.form_submit_button("Ask", type="primary")

    if ask_button:
        if not q.strip():
            st.warning("Please type a question.")
        elif not st.session_state.get("student_vec"):
            st.warning("Build your KB first.")
        else:
            with st.spinner("Thinking..."):
                chunks_scored = retrieve(
                    q,
                    st.session_state["student_vec"],
                    st.session_state["student_mat"],
                    st.session_state["student_chunks"],
                    k=6
                )

                ans = rag_answer(q, chunks_scored, llm_base_url, llm_model, llm_temp)

                st.write("**Answer**")
                st.write(ans)

                # Save chat
                save_chat(st.session_state["user_id"], q, ans)

                if chunks_scored:
                    with st.expander("📖 Context used"):
                        for i, (c, s) in enumerate(chunks_scored, 1):
                            st.markdown(f"**Chunk {i} (score {s:.3f})**\n\n{c}")

st.divider()
st.subheader("🕘 Recent chats")

rows = get_chat_history(st.session_state["user_id"], limit=10)
if not rows:
    st.caption("No history yet.")
else:
    for dt, prompt, resp in rows:
        with st.expander(f"[{dt}] {prompt[:60]}…"):
            st.write("**Q:**", prompt)
            st.write("**A:**", resp)

