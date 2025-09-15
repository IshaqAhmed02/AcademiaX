
# pages/Teacher.py
# --------------------------------------------------
# Teacher workspace: upload PDFs, build KB, generate papers, timetable
# FIXED VERSION for AcademiaX
# --------------------------------------------------

import io
import json
import pandas as pd
import streamlit as st
from teacher_functions import (
    create_teacher_tables, save_question_paper, get_question_papers,
    get_question_paper_content, save_timetable, get_timetables
)
from app_ollama_authenticated_fixed import (
    get_uploads, save_upload, pdfs_to_text_chunks,
    build_vector_store, retrieve, test_ollama_connection, call_local_llm_chat
)

# Session gate + role check
if "user_id" not in st.session_state or st.session_state["user_id"] is None:
    st.error("Please log in from the AcademiaX home page.")
    st.stop()

st.set_page_config(page_title="AcademiaX — Teacher", page_icon="👨‍🏫", layout="wide")
st.title("👨‍🏫 Teacher — Authoring & Scheduling")

create_teacher_tables()

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
for key in ["teacher_chunks", "teacher_vec", "teacher_mat"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "chunks" in key else None

tab_upload, tab_papers, tab_timetable = st.tabs(["📚 Upload & KB", "📝 Question Papers", "📅 Timetable"])

with tab_upload:
    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader("Upload teacher PDFs")

        files = st.file_uploader(
            "Upload PDF(s) (syllabus, notes, question bank…)", 
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

        if st.button("🔨 Build Teacher KB", type="primary"):
            if not uploads:
                st.warning("Upload PDFs first.")
            else:
                with st.spinner("Processing your materials..."):
                    kb_files = [io.BytesIO(content) for (name, content) in uploads]
                    chunks = pdfs_to_text_chunks(kb_files, chunk_size=900, overlap=150)

                    if not chunks:
                        st.error("Couldn't extract text (scanned PDFs?).")
                    else:
                        vec, mat = build_vector_store(chunks)
                        st.session_state["teacher_chunks"] = chunks
                        st.session_state["teacher_vec"] = vec
                        st.session_state["teacher_mat"] = mat
                        st.success(f"KB built with {len(chunks)} chunks.")

        st.metric("KB chunks", len(st.session_state.get("teacher_chunks") or []))

    with colR:
        st.subheader("Quick probe")

        with st.form("probe_form"):
            probe_q = st.text_input("Ask briefly to check your materials are indexed")
            probe_button = st.form_submit_button("Probe")

        if probe_button:
            if not probe_q.strip():
                st.warning("Type a question first.")
            elif not st.session_state.get("teacher_vec"):
                st.warning("Build your KB first.")
            else:
                with st.spinner("Searching..."):
                    chunks_scored = retrieve(
                        probe_q,
                        st.session_state["teacher_vec"],
                        st.session_state["teacher_mat"],
                        st.session_state["teacher_chunks"],
                        k=6
                    )

                    context = "\n\n".join([f"- {c}" for c, s in chunks_scored])
                    sys = "Answer ONLY using the provided context. If not present, say you don't know."
                    user = f"Question: {probe_q}\n\nContext:\n{context}"

                    ans = call_local_llm_chat(sys, user, llm_base_url, llm_model, temperature=llm_temp)

                    st.write("**Answer**")
                    st.write(ans)

with tab_papers:
    st.subheader("Generate grounded question papers")

    colA, colB = st.columns([1, 1])

    with colA:
        with st.form("paper_form"):
            title = st.text_input("Paper title", value="Unit Assessment")
            kind = st.selectbox("Paper type", ["assignment", "quiz", "midterm", "final"], index=0)
            num_q = st.number_input("Number of questions", min_value=3, max_value=50, value=10, step=1)
            focus = st.text_input("Optional topic focus (bias retrieval)", value="")

            generate_button = st.form_submit_button("🎯 Generate paper", type="primary")

        if generate_button:
            if not st.session_state.get("teacher_vec"):
                st.warning("Build your KB first.")
            else:
                with st.spinner("Generating question paper..."):
                    seed = focus or "Create a comprehensive set of questions from these materials"
                    chunks_scored = retrieve(
                        seed,
                        st.session_state["teacher_vec"],
                        st.session_state["teacher_mat"],
                        st.session_state["teacher_chunks"],
                        k=12
                    )

                    context = "\n\n".join([f"- {c}" for c, s in chunks_scored])
                    sys = "You generate high-quality exam questions strictly grounded in the context. Do NOT invent facts."
                    user = (
                        f"Create a {kind} question paper titled '{title}' with {int(num_q)} questions.\n"
                        f"Include a mix of MCQs and short answer questions.\n"
                        f"For MCQs provide 4 options and mark the correct one like [Answer: B].\n"
                        f"Keep questions clear and unambiguous.\n\nContext:\n{context}"
                    )

                    paper = call_local_llm_chat(sys, user, llm_base_url, llm_model, temperature=llm_temp)

                    if paper:
                        st.success("✅ Question paper generated!")
                        st.text_area("Preview", paper, height=420)

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save paper"):
                                save_question_paper(st.session_state["user_id"], title, kind, paper)
                                st.success("Saved!")

                        with col2:
                            st.download_button(
                                "📄 Download .txt",
                                data=paper.encode("utf-8"),
                                file_name=f"{title.replace(' ','_')}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error("Failed to generate paper. Please try again.")

    with colB:
        st.write("**Saved papers**")

        rows = get_question_papers(st.session_state["user_id"], limit=20)
        if not rows:
            st.caption("No papers saved yet.")
        else:
            for pid, ptitle, ptype, created_at in rows:
                with st.expander(f"[{created_at}] {ptitle} ({ptype})"):
                    content = get_question_paper_content(pid) or ""
                    st.text_area("Content", content, height=300, key=f"paper_content_{pid}")
                    st.download_button(
                        "Download .txt",
                        data=content.encode("utf-8"),
                        file_name=f"{ptitle.replace(' ','_')}.txt",
                        mime="text/plain",
                        key=f"download_{pid}"
                    )

with tab_timetable:
    st.subheader("Timetable generator (round-robin)")

    colL, colR = st.columns([1, 1])

    with colL:
        with st.form("timetable_form"):
            name = st.text_input("Schedule name", value="Week Plan")
            faculty = st.text_area(
                "Faculty (one per line)", 
                value="Dr. Rao\nProf. Mehta\nMs. Kapoor",
                height=100
            )
            slots = st.text_area(
                "Time slots (one per line)", 
                value="Mon 09:00-10:00\nMon 10:00-11:00\nTue 09:00-10:00\nTue 10:00-11:00",
                height=120
            )
            courses = st.text_area(
                "Optional courses (one per line)", 
                value="Math 101\nPhysics 201\nChem 110",
                height=100
            )

            generate_tt_button = st.form_submit_button("🗓️ Build schedule", type="primary")

        if generate_tt_button:
            fac = [s.strip() for s in faculty.splitlines() if s.strip()]
            sl = [s.strip() for s in slots.splitlines() if s.strip()]
            co = [s.strip() for s in courses.splitlines() if s.strip()]

            if not fac or not sl:
                st.warning("Provide at least one faculty and one time slot.")
            else:
                rows = []
                fi = 0
                ci = 0

                for slot in sl:
                    row = {"Time Slot": slot, "Faculty": fac[fi % len(fac)]}
                    if co:
                        row["Course"] = co[ci % len(co)]
                        ci += 1
                    rows.append(row)
                    fi += 1

                if rows:
                    df = pd.DataFrame(rows)
                    st.success("✅ Timetable generated!")
                    st.dataframe(df, use_container_width=True)

                    # Create CSV
                    csv = df.to_csv(index=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📄 Download CSV", 
                            data=csv.encode("utf-8"), 
                            file_name="timetable.csv", 
                            mime="text/csv"
                        )

                    with col2:
                        if st.button("💾 Save timetable"):
                            save_timetable(st.session_state["user_id"], name, json.dumps(rows))
                            st.success("Saved!")

    with colR:
        st.write("**Saved timetables**")

        saved = get_timetables(st.session_state["user_id"])
        if not saved:
            st.caption("None yet.")
        else:
            for tid, tname, created_at in saved:
                with st.expander(f"[{created_at}] {tname}"):
                    st.caption("Timetable saved successfully. Use the generator to create new ones.")

