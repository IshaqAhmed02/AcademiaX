
# teacher_page.py
# ------------------------------------------
# Complete Teacher Interface for the authenticated RAG app
# Run this as: streamlit run teacher_page.py
# CORRECTED VERSION for AcademiaX
# ------------------------------------------
import streamlit as st
import pandas as pd
import json
from io import BytesIO
from typing import List, Dict
import sqlite3

# Import from our existing files
from app_ollama_authenticated_fixed import (
    create_tables, authenticate_user, get_uploads, save_upload,
    pdfs_to_text_chunks, build_vector_store, retrieve, test_ollama_connection,
    call_local_llm_chat
)
from teacher_functions import (
    create_teacher_tables, save_question_paper, get_question_papers,
    get_question_paper_content, save_timetable, get_timetables,
    save_teacher_material, get_teacher_materials, generate_advanced_questions,
    generate_smart_timetable, analyze_curriculum_coverage, generate_lesson_plan
)

def main():
    st.set_page_config(
        page_title="👨‍🏫 AcademiaX Teacher Portal", 
        page_icon="👨‍🏫", 
        layout="wide"
    )

    # Initialize databases
    create_tables()
    create_teacher_tables()

    # Initialize session state
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
        st.session_state["username"] = None
        st.session_state["teacher_chunks"] = []
        st.session_state["teacher_vec"] = None
        st.session_state["teacher_mat"] = None

    st.title("👨‍🏫 AcademiaX Teacher Portal - Advanced Educational Tools")

    # --- AUTHENTICATION (Simplified for teacher page) ---
    if st.session_state["user_id"] is None:
        st.subheader("🔑 Teacher Login")

        with st.form("teacher_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login as Teacher")

            if login_button and username and password:
                success, user_id = authenticate_user(username, password)
                if success:
                    st.session_state["user_id"] = user_id
                    st.session_state["username"] = username
                    st.success(f"Welcome, Teacher {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        st.info("💡 Use your existing account credentials or create one in the main app first.")
        st.stop()

    # --- LOGGED IN TEACHER INTERFACE ---
    st.sidebar.success(f"👨‍🏫 Teacher: **{st.session_state['username']}**")

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

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Curriculum Materials", 
        "📝 Question Papers", 
        "📅 Timetables", 
        "📋 Lesson Plans",
        "📊 Analytics"
    ])

    # =================== TAB 1: CURRICULUM MATERIALS ===================
    with tab1:
        st.subheader("📚 Curriculum Materials Management")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**📤 Upload New Materials**")

            # Subject and topic selection
            subject = st.selectbox(
                "Subject", 
                ["Mathematics", "Physics", "Chemistry", "Biology", "English", "History", "Geography", "Computer Science", "Other"],
                index=0
            )

            if subject == "Other":
                subject = st.text_input("Enter custom subject name")

            topic = st.text_input("Topic/Chapter", placeholder="e.g., Quadratic Equations, Photosynthesis")

            # File upload
            uploaded_materials = st.file_uploader(
                "Upload curriculum materials (PDFs)", 
                accept_multiple_files=True, 
                type=["pdf"],
                help="Upload textbooks, reference materials, syllabus documents"
            )

            if uploaded_materials and subject and topic:
                for file in uploaded_materials:
                    file.seek(0)
                    save_teacher_material(st.session_state["user_id"], subject, topic, file.name, file)
                st.success(f"Uploaded {len(uploaded_materials)} files for {subject} - {topic}!")

            # Build knowledge base
            if st.button("🔨 Build Teacher Knowledge Base", type="primary"):
                all_materials = get_teacher_materials(st.session_state["user_id"])
                if all_materials:
                    with st.spinner("Processing all your materials..."):
                        kb_files = [BytesIO(content) for _, content, _, _ in all_materials]
                        chunks = pdfs_to_text_chunks(kb_files, chunk_size=1000, overlap=200)

                        if chunks:
                            vec, mat = build_vector_store(chunks)
                            st.session_state["teacher_chunks"] = chunks
                            st.session_state["teacher_vec"] = vec
                            st.session_state["teacher_mat"] = mat
                            st.success(f"✅ Built knowledge base with {len(chunks)} chunks!")
                        else:
                            st.error("❌ Could not extract text from materials.")
                else:
                    st.warning("⚠️ Please upload some materials first.")

        with col2:
            st.write("**📁 Your Materials Library**")

            # Filter by subject
            filter_subject = st.selectbox("Filter by Subject", ["All"] + ["Mathematics", "Physics", "Chemistry", "Biology", "English", "History", "Geography", "Computer Science"])

            materials = get_teacher_materials(
                st.session_state["user_id"], 
                None if filter_subject == "All" else filter_subject
            )

            if materials:
                materials_df = pd.DataFrame(materials, columns=["Filename", "Content", "Topic", "Subject"])
                materials_df = materials_df.drop("Content", axis=1)  # Don't show content in table
                st.dataframe(materials_df, use_container_width=True)

                st.metric("Total Materials", len(materials))

                # Subject distribution
                if len(materials) > 0:
                    subject_counts = pd.Series([m[3] for m in materials]).value_counts()
                    st.bar_chart(subject_counts)
            else:
                st.info("No materials uploaded yet.")

    # =================== TAB 2: QUESTION PAPERS ===================
    with tab2:
        st.subheader("📝 Advanced Question Paper Generator")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**✨ Generate New Question Paper**")

            with st.form("question_paper_form"):
                paper_title = st.text_input("Paper Title", value="Mid-Term Assessment")

                col_a, col_b = st.columns(2)
                with col_a:
                    num_questions = st.number_input("Number of Questions", min_value=5, max_value=50, value=15)
                    paper_type = st.selectbox("Paper Type", ["Assignment", "Quiz", "Mid-term", "Final", "Practice"])

                with col_b:
                    difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard", "Mixed"])
                    duration = st.number_input("Duration (minutes)", min_value=30, max_value=300, value=90)

                # Question types
                st.write("**Question Types to Include:**")
                mcq = st.checkbox("Multiple Choice Questions", value=True)
                short_ans = st.checkbox("Short Answer Questions", value=True)
                long_ans = st.checkbox("Long Answer Questions", value=True)
                fill_blanks = st.checkbox("Fill in the Blanks", value=False)
                true_false = st.checkbox("True/False", value=False)

                topic_focus = st.text_input("Topic Focus (optional)", placeholder="Leave empty for general paper")

                generate_button = st.form_submit_button("🎯 Generate Question Paper", type="primary")

            if generate_button:
                if not st.session_state.get("teacher_vec"):
                    st.error("❌ Please build your knowledge base first!")
                else:
                    # Prepare question types
                    question_types = []
                    if mcq: question_types.append("MCQ")
                    if short_ans: question_types.append("Short Answer")
                    if long_ans: question_types.append("Long Answer")
                    if fill_blanks: question_types.append("Fill in the blanks")
                    if true_false: question_types.append("True/False")

                    if not question_types:
                        st.error("Please select at least one question type!")
                    else:
                        with st.spinner("Generating your question paper..."):
                            # Retrieve relevant content
                            search_query = topic_focus if topic_focus.strip() else "curriculum syllabus educational content"
                            chunks_scored = retrieve(
                                search_query,
                                st.session_state["teacher_vec"],
                                st.session_state["teacher_mat"],
                                st.session_state["teacher_chunks"],
                                k=15
                            )

                            # Generate paper
                            paper_content = generate_advanced_questions(
                                paper_title, num_questions, chunks_scored,
                                llm_base_url, llm_model, llm_temp, paper_type.lower(),
                                difficulty.lower(), question_types
                            )

                            if paper_content:
                                # Save to database
                                save_question_paper(
                                    st.session_state["user_id"], 
                                    paper_title, 
                                    paper_type, 
                                    paper_content
                                )

                                st.success("✅ Question paper generated and saved!")

                                # Display paper
                                st.text_area("Generated Paper Preview", value=paper_content, height=400)

                                # Download button
                                st.download_button(
                                    "💾 Download Question Paper",
                                    data=paper_content,
                                    file_name=f"{paper_type}_{paper_title.replace(' ', '_')}.txt",
                                    mime="text/plain"
                                )
                            else:
                                st.error("Failed to generate question paper. Please try again.")

        with col2:
            st.write("**📋 Saved Question Papers**")

            saved_papers = get_question_papers(st.session_state["user_id"], limit=20)

            if saved_papers:
                for paper_id, title, paper_type, created_at in saved_papers:
                    with st.expander(f"📄 {title} ({paper_type}) - {created_at}"):
                        if st.button(f"View Full Paper", key=f"view_{paper_id}"):
                            content = get_question_paper_content(paper_id)
                            if content:
                                st.text_area("Paper Content", value=content, height=300, key=f"content_{paper_id}")
                                st.download_button(
                                    "Download",
                                    data=content,
                                    file_name=f"{title}.txt",
                                    mime="text/plain",
                                    key=f"download_{paper_id}"
                                )
            else:
                st.info("No saved question papers yet.")

    # =================== TAB 3: TIMETABLES ===================
    with tab3:
        st.subheader("📅 Smart Timetable Generator")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**🗓️ Create New Timetable**")

            with st.form("timetable_form"):
                timetable_name = st.text_input("Timetable Name", value="Weekly Schedule")

                # Faculty input
                faculty_input = st.text_area(
                    "Faculty Members (one per line)",
                    value="Dr. Smith (Mathematics)\nProf. Johnson (Physics)\nMs. Davis (Chemistry)\nDr. Brown (Biology)",
                    height=100
                )

                # Time slots
                slots_input = st.text_area(
                    "Time Slots (one per line)",
                    value="Monday 09:00-10:00\nMonday 10:00-11:00\nMonday 11:15-12:15\nTuesday 09:00-10:00\nTuesday 10:00-11:00\nWednesday 09:00-10:00",
                    height=120
                )

                # Subjects
                subjects_input = st.text_area(
                    "Subjects (one per line)",
                    value="Mathematics\nPhysics\nChemistry\nBiology\nEnglish\nHistory",
                    height=100
                )

                # Constraints
                st.write("**Constraints (Optional):**")
                max_hours = st.number_input("Max hours per faculty per day", min_value=1, max_value=8, value=4)

                generate_tt_button = st.form_submit_button("🔄 Generate Timetable", type="primary")

            if generate_tt_button:
                faculty_list = [f.strip() for f in faculty_input.strip().split("\n") if f.strip()]
                slots_list = [s.strip() for s in slots_input.strip().split("\n") if s.strip()]
                subjects_list = [s.strip() for s in subjects_input.strip().split("\n") if s.strip()]

                if not faculty_list or not slots_list:
                    st.error("❌ Please provide faculty and time slots!")
                else:
                    constraints = {"max_hours_per_faculty": max_hours}

                    timetable_data = generate_smart_timetable(
                        faculty_list, slots_list, subjects_list, constraints
                    )

                    if timetable_data:
                        df = pd.DataFrame(timetable_data)

                        # Save to database
                        save_timetable(
                            st.session_state["user_id"], 
                            timetable_name, 
                            json.dumps(timetable_data)
                        )

                        st.success("✅ Timetable generated and saved!")
                        st.dataframe(df, use_container_width=True)

                        # Download options
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "💾 Download as CSV",
                            data=csv_data,
                            file_name=f"{timetable_name.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )

        with col2:
            st.write("**📊 Saved Timetables**")

            saved_timetables = get_timetables(st.session_state["user_id"])

            if saved_timetables:
                for tt_id, name, created_at in saved_timetables:
                    with st.expander(f"📅 {name} - {created_at}"):
                        st.write(f"Timetable ID: {tt_id}")
                        st.caption("Timetable saved successfully.")
            else:
                st.info("No saved timetables yet.")

    # =================== TAB 4: LESSON PLANS ===================
    with tab4:
        st.subheader("📋 AI-Powered Lesson Plan Generator")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**✨ Create Lesson Plan**")

            with st.form("lesson_plan_form"):
                lesson_topic = st.text_input("Lesson Topic", placeholder="e.g., Introduction to Photosynthesis")
                lesson_duration = st.selectbox("Duration", [30, 45, 60, 90, 120], index=2)
                grade_level = st.selectbox("Grade Level", ["6th", "7th", "8th", "9th", "10th", "11th", "12th"])

                learning_style = st.selectbox(
                    "Teaching Approach",
                    ["Interactive", "Lecture-based", "Hands-on", "Discussion-based", "Mixed"]
                )

                special_requirements = st.text_area(
                    "Special Requirements (optional)",
                    placeholder="e.g., Lab equipment needed, group activities, multimedia"
                )

                generate_lp_button = st.form_submit_button("🎯 Generate Lesson Plan", type="primary")

            if generate_lp_button and lesson_topic:
                if not st.session_state.get("teacher_vec"):
                    st.error("❌ Please build your knowledge base first!")
                else:
                    with st.spinner("Creating your lesson plan..."):
                        # Retrieve relevant content for the topic
                        chunks_scored = retrieve(
                            lesson_topic,
                            st.session_state["teacher_vec"],
                            st.session_state["teacher_mat"],
                            st.session_state["teacher_chunks"],
                            k=8
                        )

                        lesson_plan = generate_lesson_plan(
                            lesson_topic, lesson_duration, chunks_scored,
                            llm_base_url, llm_model, llm_temp
                        )

                        if lesson_plan:
                            st.success("✅ Lesson plan generated!")
                            st.text_area("Generated Lesson Plan", value=lesson_plan, height=500)

                            st.download_button(
                                "💾 Download Lesson Plan",
                                data=lesson_plan,
                                file_name=f"lesson_plan_{lesson_topic.replace(' ', '_')}.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("Failed to generate lesson plan.")

        with col2:
            st.write("**💡 Lesson Plan Tips**")
            st.info("""
            **Good lesson plans include:**
            - Clear learning objectives
            - Engaging introduction
            - Well-structured content delivery
            - Interactive activities
            - Assessment methods
            - Time management
            - Differentiated instruction
            """)

            st.write("**🎯 Quick Topic Generator**")
            if st.button("🎲 Suggest Random Topic"):
                topics = [
                    "Ecosystem and Food Chains", "Quadratic Equations", "World War History",
                    "Chemical Reactions", "Literature Analysis", "Geometry Theorems",
                    "Cell Division", "Grammar Rules", "Solar System", "Probability"
                ]
                import random
                suggested = random.choice(topics)
                st.write(f"💡 Suggested topic: **{suggested}**")

    # =================== TAB 5: ANALYTICS ===================
    with tab5:
        st.subheader("📊 Teaching Analytics & Insights")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**📈 Your Teaching Statistics**")

            # Get statistics
            question_papers = get_question_papers(st.session_state["user_id"], limit=1000)
            timetables = get_timetables(st.session_state["user_id"])
            materials = get_teacher_materials(st.session_state["user_id"])

            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Question Papers", len(question_papers))
            with col_b:
                st.metric("Timetables Created", len(timetables))
            with col_c:
                st.metric("Materials Uploaded", len(materials))

            # Subject distribution
            if materials:
                subjects = [m[3] for m in materials]  # Subject is 4th column
                subject_df = pd.Series(subjects).value_counts()
                st.write("**📚 Materials by Subject**")
                st.bar_chart(subject_df)

            # Paper types distribution
            if question_papers:
                paper_types = [p[2] for p in question_papers]  # Paper type is 3rd column
                type_df = pd.Series(paper_types).value_counts()
                st.write("**📝 Question Papers by Type**")
                st.bar_chart(type_df)

        with col2:
            st.write("**🎯 Curriculum Coverage Analysis**")

            # Sample syllabus topics for demonstration
            sample_topics = st.multiselect(
                "Select syllabus topics to analyze coverage:",
                ["Algebra", "Geometry", "Calculus", "Statistics", "Trigonometry", 
                 "Physics", "Chemistry", "Biology", "History", "Geography"],
                default=["Algebra", "Geometry"]
            )

            if sample_topics and st.session_state.get("teacher_chunks"):
                coverage = analyze_curriculum_coverage(
                    st.session_state["teacher_chunks"], 
                    sample_topics
                )

                coverage_df = pd.DataFrame([
                    {
                        "Topic": topic,
                        "Mentions": data["mentions"],
                        "Covered": "✅" if data["covered"] else "❌",
                        "Coverage %": data["coverage_score"]
                    }
                    for topic, data in coverage.items()
                ])

                st.dataframe(coverage_df, use_container_width=True)

                # Coverage chart
                st.bar_chart(coverage_df.set_index("Topic")["Coverage %"])
            else:
                st.info("Upload materials and select topics to see coverage analysis.")

            st.write("**💡 Recommendations**")
            st.success("✅ Well covered topics: Focus on advanced questions")
            st.warning("⚠️ Partially covered: Add more reference materials")
            st.error("❌ Not covered: Upload relevant content")

if __name__ == "__main__":
    main()

