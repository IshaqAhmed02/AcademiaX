
# main_fixed.py
# ------------------------------------------
# AcademiaX - Main Landing Page with Database Migration Fix
# Run with: streamlit run main_fixed.py
# ------------------------------------------
import streamlit as st
import sqlite3
import hashlib
from typing import Tuple
import subprocess
import sys
import os

# Database setup
DB_PATH = "academiax.db"

def migrate_database():
    """Handle database migrations for existing installations"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        # Check if users table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not c.fetchone():
            # Create fresh tables
            create_main_tables()
            return

        # Check and add missing columns
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]

        if 'is_active' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
        if 'role' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'student'")
        if 'full_name' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
        if 'email' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN email TEXT")
        if 'created_at' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        if 'last_login' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP")

        conn.commit()

def create_main_tables():
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
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                grade_level TEXT,
                subjects TEXT,
                institution TEXT,
                bio TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS login_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
        conn.commit()

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, email: str, password: str, role: str, full_name: str = "") -> Tuple[bool, str]:
    """Register a new user with role"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, email, password_hash, role, full_name) VALUES (?, ?, ?, ?, ?)",
                (username, email or f"{username}@example.com", hash_password(password), role, full_name)
            )
            conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists!"
        elif "email" in str(e):
            return False, "Email already registered!"
        else:
            return False, "Registration failed!"
    except Exception as e:
        return False, f"Error: {str(e)}"

def authenticate_user(username: str, password: str) -> Tuple[bool, int, str, str]:
    """Authenticate user and return user details"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, role, full_name, email FROM users WHERE username=? AND password_hash=?",
            (username, hash_password(password))
        )
        result = c.fetchone()
        if result:
            user_id, role, full_name, email = result
            role = role or 'student'  # Default role if null
            # Update last login
            c.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user_id,))
            conn.commit()
            return True, user_id, role, full_name or username
        return False, None, None, None

def get_user_stats() -> dict:
    """Get platform statistics with safe column access"""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        try:
            # Total users
            c.execute("SELECT COUNT(*) FROM users")
            total_users = c.fetchone()[0]

            # Students and teachers
            c.execute("SELECT COUNT(*) FROM users WHERE role='student'")
            total_students = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM users WHERE role='teacher'")
            total_teachers = c.fetchone()[0]

            # Recent registrations (last 7 days)
            c.execute("SELECT COUNT(*) FROM users WHERE created_at >= datetime('now', '-7 days')")
            recent_registrations = c.fetchone()[0]

        except sqlite3.OperationalError:
            # Fallback if columns don't exist
            c.execute("SELECT COUNT(*) FROM users")
            total_users = c.fetchone()[0]
            total_students = max(0, total_users // 2)  # Estimate
            total_teachers = total_users - total_students
            recent_registrations = min(5, total_users)  # Estimate

        return {
            "total_users": total_users,
            "total_students": total_students,
            "total_teachers": total_teachers,
            "recent_registrations": recent_registrations
        }

def launch_student_app():
    """Launch student application on port 8503"""
    try:
        if os.path.exists("app_ollama_authenticated_fixed.py"):
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app_ollama_authenticated_fixed.py", "--server.port", "8503"])
            return True, "Student app launching on port 8503..."
        else:
            return False, "Student app file not found!"
    except Exception as e:
        return False, f"Error launching student app: {str(e)}"

def launch_teacher_app():
    """Launch teacher application on port 8504"""
    try:
        if os.path.exists("teacher_page.py"):
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "teacher_page.py", "--server.port", "8504"])
            return True, "Teacher app launching on port 8504..."
        else:
            return False, "Teacher app file not found!"
    except Exception as e:
        return False, f"Error launching teacher app: {str(e)}"

def main():
    # Page configuration
    st.set_page_config(
        page_title="AcademiaX - AI-Powered Education Platform",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize and migrate database
    migrate_database()

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.full_name = None

    # Custom CSS for modern UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-subtitle {
        font-size: 1.5rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: #5682B1;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stats-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .success-box {
        background: linear-gradient(45deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🎓 AcademiaX</div>
        <div class="main-subtitle">AI-Powered Education Platform</div>
        <p>Revolutionizing learning with intelligent tutoring, automated assessment, and personalized education</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if user is authenticated
    if not st.session_state.authenticated:
        show_landing_page()
    else:
        show_dashboard()

def show_landing_page():
    """Show the main landing page with authentication"""

    # Platform statistics
    stats = get_user_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h3>{stats['total_users']}</h3>
            <p>Total Users</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <h3>{stats['total_students']}</h3>
            <p>Students</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <h3>{stats['total_teachers']}</h3>
            <p>Teachers</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <h3>{stats['recent_registrations']}</h3>
            <p>New This Week</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>🎓 For Students</h2>
            <ul>
                <li><strong>AI-Powered Chat:</strong> Ask questions about your study materials</li>
                <li><strong>Smart Document Analysis:</strong> Upload PDFs and get instant insights</li>
                <li><strong>Personalized Learning:</strong> Adaptive content based on your progress</li>
                <li><strong>Study History:</strong> Track your learning journey</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h2>👨‍🏫 For Teachers</h2>
            <ul>
                <li><strong>Question Paper Generation:</strong> Create exams automatically</li>
                <li><strong>Curriculum Management:</strong> Organize materials by subject</li>
                <li><strong>Smart Timetabling:</strong> Optimize class schedules</li>
                <li><strong>Lesson Plan Creation:</strong> AI-generated structured lessons</li>
                <li><strong>Analytics Dashboard:</strong> Track teaching effectiveness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Authentication section
        st.subheader("🔐 Get Started")

        auth_tab1, auth_tab2 = st.tabs(["🚪 Login", "📝 Sign Up"])

        with auth_tab1:
            with st.form("login_form"):
                st.markdown("**Welcome Back! Please sign in to continue.**")

                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")

                login_button = st.form_submit_button("🔑 Login", type="primary", use_container_width=True)

                if login_button and username and password:
                    success, user_id, role, full_name = authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.session_state.role = role
                        st.session_state.full_name = full_name
                        st.success(f"Welcome back, {full_name}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")

        with auth_tab2:
            with st.form("signup_form"):
                st.markdown("**Join AcademiaX! Create your account.**")

                role = st.selectbox("I am a:", ["student", "teacher"], format_func=lambda x: f"🎓 Student" if x == "student" else "👨‍🏫 Teacher")

                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email (Optional)", placeholder="Enter your email address")
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Choose a strong password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

                terms_agreed = st.checkbox("I agree to the Terms of Service and Privacy Policy")

                signup_button = st.form_submit_button("🎯 Create Account", type="primary", use_container_width=True)

                if signup_button:
                    # Validation
                    if not all([full_name, new_username, new_password, confirm_password]):
                        st.error("❌ Please fill in all required fields.")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords don't match!")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters long.")
                    elif not terms_agreed:
                        st.error("❌ Please agree to the terms and conditions.")
                    else:
                        success, message = register_user(new_username, email, new_password, role, full_name)
                        if success:
                            st.success("✅ Registration successful! Please login with your credentials.")
                        else:
                            st.error(f"❌ {message}")

def show_dashboard():
    """Show post-login dashboard with role-based options"""

    # Header with user info
    st.markdown(f"""
    <div class="success-box">
        <h2>Welcome back, {st.session_state.full_name}! 👋</h2>
        <p>Role: {st.session_state.role.title()} | Username: {st.session_state.username}</p>
    </div>
    """, unsafe_allow_html=True)

    # Logout button in sidebar
    with st.sidebar:
        st.write(f"**Logged in as:** {st.session_state.full_name}")
        st.write(f"**Role:** {st.session_state.role.title()}")

        if st.button("🚪 Logout", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown("## 🎯 Choose Your Learning Experience")

    col1, col2 = st.columns(2)

    if st.session_state.role == "student":
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h2>🎓 Student Portal</h2>
                <p>Access your personalized learning environment with:</p>
                <ul>
                    <li>AI-powered chat with your study materials</li>
                    <li>Document upload and analysis</li>
                    <li>Learning progress tracking</li>
                    <li>Personalized study recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Launch Student App", type="primary", use_container_width=True):
                success, message = launch_student_app()
                if success:
                    st.success(message)
                    st.info("🌐 Your student app will open in a new tab at: http://localhost:8503")
                    st.markdown("**[Click here to open Student Portal](http://localhost:8503)**", unsafe_allow_html=True)
                else:
                    st.error(message)

        with col2:
            st.markdown("### 📊 Your Quick Stats")
            st.metric("Account Type", "Student")
            st.metric("Status", "Active")

            st.markdown("### 🎯 Getting Started")
            st.markdown("""
            1. **Upload Materials**: Add your textbooks and notes
            2. **Ask Questions**: Chat with AI about your content  
            3. **Track Progress**: Monitor your learning journey
            4. **Get Help**: Use AI tutoring features
            """)

    elif st.session_state.role == "teacher":
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h2>👨‍🏫 Teacher Portal</h2>
                <p>Access your comprehensive teaching toolkit with:</p>
                <ul>
                    <li>Automated question paper generation</li>
                    <li>Smart timetable creation</li>
                    <li>Curriculum material management</li>
                    <li>AI-powered lesson planning</li>
                    <li>Teaching analytics and insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Launch Teacher Portal", type="primary", use_container_width=True):
                success, message = launch_teacher_app()
                if success:
                    st.success(message)
                    st.info("🌐 Your teacher portal will open in a new tab at: http://localhost:8504")
                    st.markdown("**[Click here to open Teacher Portal](http://localhost:8504)**", unsafe_allow_html=True)
                else:
                    st.error(message)

        with col2:
            st.markdown("### 📊 Your Quick Stats")
            st.metric("Account Type", "Teacher")
            st.metric("Status", "Active")

            st.markdown("### 🎯 Getting Started")
            st.markdown("""
            1. **Upload Curriculum**: Add your teaching materials
            2. **Create Papers**: Generate question papers automatically
            3. **Plan Lessons**: Use AI for structured lesson plans
            4. **Track Analytics**: Monitor teaching effectiveness
            """)

if __name__ == "__main__":
    main()

