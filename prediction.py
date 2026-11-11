import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from database import get_connection

MODEL_PATH = "absence_model.pkl"


def prepare_training_data():
    """
    Create training data from the attendance table.
    Features:
    - day_of_week (0=Monday ... 6=Sunday)
    - attended_last_7_days
    - total_absences_so_far
    Target:
    - will_be_absent_tomorrow (1 = absent, 0 = present)
    """
    conn = get_connection()

    # Get all attendance records
    df = pd.read_sql_query("""
        SELECT name, timestamp 
        FROM attendance 
        ORDER BY name, timestamp
    """, conn)
    conn.close()

    if df.empty:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Get unique people and date range
    people = df['name'].unique()
    min_date = df['date'].min()
    max_date = df['date'].max()

    records = []

    for person in people:
        person_dates = set(df[df['name'] == person]['date'])

        current = min_date
        while current <= max_date - timedelta(days=1):
            # Features
            day_of_week = current.weekday()  # 0 = Monday

            # How many days the person attended in the last 7 days
            last_7_start = current - timedelta(days=7)
            attended_last_7 = sum(
                1 for d in person_dates 
                if last_7_start <= d < current
            )

            # Total absences so far (rough calculation)
            total_days_so_far = (current - min_date).days + 1
            total_present = sum(1 for d in person_dates if d <= current)
            total_absences = max(0, total_days_so_far - total_present)

            # Target: was the person absent the next day?
            next_day = current + timedelta(days=1)
            was_absent = 0 if next_day in person_dates else 1

            records.append({
                "day_of_week": day_of_week,
                "attended_last_7": attended_last_7,
                "total_absences": total_absences,
                "absent_next_day": was_absent
            })

            current += timedelta(days=1)

    data = pd.DataFrame(records)
    return data


def train_absence_model():
    """Train and save the absence prediction model"""
    data = prepare_training_data()

    if data is None or len(data) < 20:
        return False, "Not enough attendance data to train the model (need more records)."

    X = data[["day_of_week", "attended_last_7", "total_absences"]]
    y = data["absent_next_day"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model
    joblib.dump(model, MODEL_PATH)

    return True, f"Model trained successfully! Accuracy: {accuracy:.2%}"


def predict_absence_risk(name):
    """
    Predict if a student is likely to be absent tomorrow.
    Returns: "High", "Medium", or "Low"
    """
    if not os.path.exists(MODEL_PATH):
        return "Model not trained yet"

    model = joblib.load(MODEL_PATH)
    conn = get_connection()

    # Get this person's attendance history
    df = pd.read_sql_query("""
        SELECT timestamp FROM attendance 
        WHERE name = ?
        ORDER BY timestamp
    """, conn, params=(name,))
    conn.close()

    if df.empty:
        return "No history"

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    person_dates = set(df['date'])

    today = datetime.now().date()

    # Features for today
    day_of_week = today.weekday()

    last_7_start = today - timedelta(days=7)
    attended_last_7 = sum(1 for d in person_dates if last_7_start <= d < today)

    total_days = (today - min(person_dates)).days + 1 if person_dates else 1
    total_present = len(person_dates)
    total_absences = max(0, total_days - total_present)

    features = np.array([[day_of_week, attended_last_7, total_absences]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # probability of absence

    if probability >= 0.65:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"