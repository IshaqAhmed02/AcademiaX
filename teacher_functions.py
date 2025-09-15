
# teacher_functions.py
# ------------------------------------------
# Teacher-specific functions for the authenticated RAG app
# CORRECTED VERSION for AcademiaX
# ------------------------------------------
import sqlite3
import re
import random
from typing import List, Tuple, Dict
import pandas as pd
import streamlit as st
from io import BytesIO

DB_PATH = "academiax.db"

def create_teacher_tables():
    """Create teacher-specific tables in the database"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS question_papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                paper_type TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS timetables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS teacher_materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                subject TEXT,
                topic TEXT,
                filename TEXT,
                content BLOB,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
        conn.commit()

def save_question_paper(user_id: int, title: str, paper_type: str, content: str):
    """Save generated question paper to database"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO question_papers (user_id, title, paper_type, content) VALUES (?, ?, ?, ?)",
            (user_id, title, paper_type, content))
        conn.commit()

def get_question_papers(user_id: int, limit: int = 20):
    """Get saved question papers for a user"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, title, paper_type, created_at FROM question_papers WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        )
        return c.fetchall()

def get_question_paper_content(paper_id: int):
    """Get content of a specific question paper"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT content FROM question_papers WHERE id=?", (paper_id,))
        result = c.fetchone()
        return result[0] if result else None

def save_timetable(user_id: int, name: str, data_json: str):
    """Save timetable to database"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO timetables (user_id, name, data) VALUES (?, ?, ?)",
            (user_id, name, data_json))
        conn.commit()

def get_timetables(user_id: int):
    """Get saved timetables for a user"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, name, created_at FROM timetables WHERE user_id=? ORDER BY id DESC",
            (user_id,)
        )
        return c.fetchall()

def save_teacher_material(user_id: int, subject: str, topic: str, filename: str, content: BytesIO):
    """Save teacher material to database"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO teacher_materials (user_id, subject, topic, filename, content) VALUES (?, ?, ?, ?, ?)",
            (user_id, subject, topic, filename, content.read()))
        conn.commit()

def get_teacher_materials(user_id: int, subject: str = None):
    """Get teacher materials, optionally filtered by subject"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        if subject:
            c.execute(
                "SELECT filename, content, topic FROM teacher_materials WHERE user_id=? AND subject=? ORDER BY upload_date DESC",
                (user_id, subject)
            )
        else:
            c.execute(
                "SELECT filename, content, topic, subject FROM teacher_materials WHERE user_id=? ORDER BY upload_date DESC",
                (user_id,)
            )
        return c.fetchall()

def generate_advanced_questions(
    title: str, 
    num_q: int, 
    chunks_scored: List[Tuple[str, float]], 
    llm_url: str, 
    llm_model: str, 
    temp: float, 
    style: str,
    difficulty: str = "medium",
    question_types: List[str] = None
) -> str:
    """Generate advanced question papers with more customization"""
    if not chunks_scored:
        return f"Could not generate questions - no relevant content found."

    # Prepare context
    context_parts = []
    total_length = 0
    max_context_length = 8000

    for chunk, score in chunks_scored:
        if total_length + len(chunk) <= max_context_length:
            context_parts.append(chunk)
            total_length += len(chunk)
        else:
            break

    context = "\n\n".join(context_parts)

    # Define question types if not provided
    if not question_types:
        question_types = ["MCQ", "Short Answer", "Long Answer", "Fill in the blanks"]

    system_prompt = (
        f"You are an expert educator creating a {difficulty} difficulty {style} exam. "
        f"Create exactly {num_q} questions based strictly on the provided content. "
        f"Use these question types: {', '.join(question_types)}. "
        "Ensure questions test different cognitive levels: knowledge, comprehension, application, and analysis."
    )

    user_prompt = (
        f"Create a {style} question paper titled '{title}' with exactly {num_q} questions.\n"
        f"Requirements:\n"
        f"- Difficulty level: {difficulty}\n"
        f"- Question types to include: {', '.join(question_types)}\n"
        f"- Number each question clearly (1, 2, 3, etc.)\n"
        f"- For MCQs: Provide 4 options (A, B, C, D) and mark correct answer as [Answer: X]\n"
        f"- For short answers: 2-3 sentences expected\n"
        f"- For long answers: 1-2 paragraphs expected\n"
        f"- Include marks allocation for each question\n"
        f"- Base all questions strictly on the provided context\n\n"
        f"Context:\n{context}"
    )

    from app_ollama_authenticated_fixed import call_local_llm_chat

    paper = call_local_llm_chat(system_prompt, user_prompt, llm_url, llm_model, temperature=temp)

    if paper:
        return f"{title}\n{'-'*50}\n{style.capitalize()} Paper - {difficulty.capitalize()} Level\n{'-'*50}\n\n{paper}"

    # Fallback question generation
    if not paper:
        st.warning("LLM error during question generation - using fallback")

    # Create fallback questions from context
    topics = re.findall(r"[A-Z][A-Za-z0-9 ]{10,80}[.!?]", context)
    if not topics:
        topics = context.split(".")[:-1]

    questions = []
    for i in range(1, min(num_q + 1, len(topics) + 1)):
        topic = topics[(i-1) % len(topics)].strip()
        q_type = question_types[i % len(question_types)]

        if q_type == "MCQ":
            questions.append(
                f"{i}) Which of the following best describes {topic[:30]}...? [2 marks]\n"
                f"   A) Option A\n   B) Option B\n   C) Option C\n   D) Option D\n"
                f"   [Answer: B]"
            )
        elif q_type == "Short Answer":
            questions.append(f"{i}) Briefly explain: {topic[:50]}... [3 marks]")
        elif q_type == "Long Answer":
            questions.append(f"{i}) Discuss in detail: {topic[:40]}... [5 marks]")
        else:  # Fill in the blanks
            questions.append(f"{i}) Fill in the blanks related to: {topic[:40]}... [2 marks]")

    return f"{title}\n{'-'*50}\n{style.capitalize()} Paper - {difficulty.capitalize()} Level\n{'-'*50}\n\n" + "\n\n".join(questions)

def generate_smart_timetable(
    faculty_list: List[str], 
    time_slots: List[str], 
    subjects: List[str] = None,
    constraints: Dict = None
) -> List[Dict]:
    """Generate intelligent timetable with constraints"""
    if not faculty_list or not time_slots:
        return []

    subjects = subjects or []
    constraints = constraints or {}

    # Parse time slots to understand days and times
    schedule = []
    faculty_workload = {f: 0 for f in faculty_list}

    for slot_idx, slot in enumerate(time_slots):
        # Try to parse day and time from slot
        slot_parts = slot.strip().split()
        day = slot_parts[0] if slot_parts else f"Day{slot_idx+1}"
        time = " ".join(slot_parts[1:]) if len(slot_parts) > 1 else f"Period {slot_idx+1}"

        # Select faculty with least workload
        available_faculty = [f for f in faculty_list if faculty_workload[f] == min(faculty_workload.values())]
        selected_faculty = available_faculty[slot_idx % len(available_faculty)]

        # Select subject
        selected_subject = subjects[slot_idx % len(subjects)] if subjects else f"Subject {slot_idx+1}"

        # Apply constraints if any
        if constraints.get("max_hours_per_faculty"):
            if faculty_workload[selected_faculty] >= constraints["max_hours_per_faculty"]:
                continue

        entry = {
            "Day": day,
            "Time": time,
            "Subject": selected_subject,
            "Faculty": selected_faculty,
            "Room": f"Room {(slot_idx % 10) + 1}"  # Simple room assignment
        }

        schedule.append(entry)
        faculty_workload[selected_faculty] += 1

    return schedule

def analyze_curriculum_coverage(chunks: List[str], syllabus_topics: List[str]) -> Dict:
    """Analyze how well uploaded materials cover syllabus topics"""
    coverage = {}
    total_content = " ".join(chunks).lower()

    for topic in syllabus_topics:
        topic_lower = topic.lower()
        # Simple keyword matching - can be enhanced with semantic similarity
        mentions = total_content.count(topic_lower)
        coverage[topic] = {
            "mentions": mentions,
            "covered": mentions > 0,
            "coverage_score": min(mentions * 10, 100)  # Cap at 100%
        }

    return coverage

def generate_lesson_plan(
    topic: str,
    duration: int,
    chunks_scored: List[Tuple[str, float]],
    llm_url: str,
    llm_model: str,
    temp: float
) -> str:
    """Generate a structured lesson plan"""
    if not chunks_scored:
        return f"Could not generate lesson plan - no relevant content found for {topic}."

    context = "\n\n".join([chunk for chunk, _ in chunks_scored[:5]])

    system_prompt = (
        "You are an expert curriculum designer. Create a detailed, structured lesson plan "
        "that is engaging, pedagogically sound, and follows best teaching practices."
    )

    user_prompt = (
        f"Create a {duration}-minute lesson plan on '{topic}'.\n"
        f"Structure:\n"
        f"1. Learning Objectives (2-3 specific, measurable objectives)\n"
        f"2. Materials Needed\n"
        f"3. Lesson Structure with timing:\n"
        f"   - Introduction/Hook (5-10 min)\n"
        f"   - Main Content (70% of time)\n"
        f"   - Activities/Practice (15-20 min)\n"
        f"   - Conclusion/Summary (5 min)\n"
        f"4. Assessment Methods\n"
        f"5. Homework/Extension Activities\n\n"
        f"Base the content on this context:\n{context[:4000]}"
    )

    from app_ollama_authenticated_fixed import call_local_llm_chat
    lesson_plan = call_local_llm_chat(system_prompt, user_prompt, llm_url, llm_model, temperature=temp)

    return lesson_plan or f"Could not generate lesson plan for {topic}. Please try again or check your LLM connection."

