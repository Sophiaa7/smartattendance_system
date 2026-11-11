import sqlite3
from datetime import datetime, timedelta
import os

DB_NAME = "attendance.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist"""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')

    # 3. Users table (for admin login)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create default admin account
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("admin", "admin123")
        )

    conn.commit()
    conn.close()
    print("✅ Database created successfully!")
    print("   → Tables: students, attendance, users")
    print("   → Default admin → username: admin | password: admin123")


# ====================== ATTENDANCE FUNCTIONS ======================

def mark_attendance(name, confidence=None):
    if not name or name.lower() == "unknown":
        return False

    conn = get_connection()
    cursor = conn.cursor()

    # Prevent duplicate within 45 seconds
    time_limit = datetime.now() - timedelta(seconds=45)
    cursor.execute('''
        SELECT id FROM attendance 
        WHERE name = ? AND timestamp > ?
        LIMIT 1
    ''', (name, time_limit.strftime("%Y-%m-%d %H:%M:%S")))

    if cursor.fetchone():
        conn.close()
        print(f"⏳ {name} already marked recently")
        return False

    # Check if student exists, otherwise add them
    cursor.execute("SELECT id FROM students WHERE name = ?", (name,))
    student = cursor.fetchone()

    if student:
        student_id = student["id"]
    else:
        cursor.execute("INSERT INTO students (name) VALUES (?)", (name,))
        student_id = cursor.lastrowid

    # Save attendance
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO attendance (student_id, name, timestamp, confidence)
        VALUES (?, ?, ?, ?)
    ''', (student_id, name, now, confidence))

    conn.commit()
    conn.close()
    print(f"✅ Attendance marked: {name}")
    return True


def get_attendance_by_date(date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, timestamp, confidence 
        FROM attendance 
        WHERE date(timestamp) = ?
        ORDER BY timestamp DESC
    ''', (date_str,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_attendance_by_person(name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, timestamp, confidence 
        FROM attendance 
        WHERE name = ?
        ORDER BY timestamp DESC
    ''', (name,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_attendance(limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, timestamp, confidence 
        FROM attendance 
        ORDER BY timestamp DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_students():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created_at FROM students ORDER BY name")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_attendance_stats():
    conn = get_connection()
    cursor = conn.cursor()

    # Total records
    cursor.execute("SELECT COUNT(*) as total FROM attendance")
    total = cursor.fetchone()["total"]

    # Unique people
    cursor.execute("SELECT COUNT(DISTINCT name) as unique_people FROM attendance")
    unique_people = cursor.fetchone()["unique_people"]

    # Today's attendance
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) as today_count FROM attendance WHERE date(timestamp) = ?", (today,))
    today_count = cursor.fetchone()["today_count"]

    # Attendance per day (last 7 days)
    cursor.execute('''
        SELECT date(timestamp) as day, COUNT(*) as count
        FROM attendance
        GROUP BY date(timestamp)
        ORDER BY day DESC
        LIMIT 7
    ''')
    daily = [dict(row) for row in cursor.fetchall()]

    # Top attendees
    cursor.execute('''
        SELECT name, COUNT(*) as count
        FROM attendance
        GROUP BY name
        ORDER BY count DESC
        LIMIT 5
    ''')
    top_people = [dict(row) for row in cursor.fetchall()]

    # Low attendance students
    cursor.execute('''
        SELECT name, COUNT(*) as count
        FROM attendance
        GROUP BY name
        HAVING count < 3
        ORDER BY count ASC
        LIMIT 5
    ''')
    low_attendance = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {
        "total_records": total,
        "unique_people": unique_people,
        "today_count": today_count,
        "daily": daily,
        "top_people": top_people,
        "low_attendance": low_attendance
    }

def add_student(name):
    """Add a new student to the database"""
    if not name or name.strip() == "":
        return False, "Name cannot be empty"

    name = name.strip()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO students (name) VALUES (?)", (name,))
        conn.commit()
        conn.close()
        return True, f"Student '{name}' added successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"Student '{name}' already exists"

def clean_attendance_data():
    """
    Clean the attendance table:
    - Remove exact duplicate records (same name + same timestamp)
    - Normalize names (capitalize properly)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Remove exact duplicates
    cursor.execute('''
        DELETE FROM attendance
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM attendance
            GROUP BY name, timestamp
        )
    ''')
    deleted_duplicates = cursor.rowcount

    # 2. Normalize names (Title Case)
    cursor.execute("SELECT DISTINCT name FROM attendance")
    names = cursor.fetchall()

    for row in names:
        old_name = row["name"]
        new_name = old_name.strip().title()   # e.g. "yinka" → "Yinka"

        if old_name != new_name:
            cursor.execute(
                "UPDATE attendance SET name = ? WHERE name = ?",
                (new_name, old_name)
            )
            # Also update students table
            cursor.execute(
                "UPDATE students SET name = ? WHERE name = ?",
                (new_name, old_name)
            )

    conn.commit()
    conn.close()

    return {
        "duplicates_removed": deleted_duplicates,
        "message": "Data cleaning completed"
    }


def get_low_attendance_students(threshold=3):
    """Return students with very few attendance records"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT name, COUNT(*) as count
        FROM attendance
        GROUP BY name
        HAVING count < ?
        ORDER BY count ASC
    ''', (threshold,))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]