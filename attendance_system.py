from flask import Flask, render_template, Response, redirect, url_for, request, session
from functools import wraps
import cv2
import numpy as np
import time
import os
from datetime import datetime
import pandas as pd
import base64

from database import (
    mark_attendance as db_mark_attendance,
    init_db,
    get_all_attendance,
    get_attendance_stats,
    add_student as db_add_student
)
import base64

app = Flask(__name__)
app.secret_key = "smart_attendance_secret_key_2026"

# Load cascade at startup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# ====================== GLOBALS ======================
model = None
labels = None
face_cascade = None
video = None

attendance_marked = False
recognized_person = None
stop_start_time = None


# ====================== LOGIN PROTECTION ======================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ====================== MODEL & CAMERA ======================
def load_model():
    global model, labels, face_cascade, video
    print("🔄 Loading face recognition model...")

    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Only open camera if it is not already open
        if video is None or not video.isOpened():
            video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not video.isOpened():
                print("❌ Camera failed to open")
                return False

        if os.path.exists("face_model.yml") and os.path.exists("labels.npy"):
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read("face_model.yml")
            labels = np.load("labels.npy", allow_pickle=True).item()
            print("✅ Model and Camera loaded!")
        else:
            print("⚠️ No model found. Running camera only.")

        return True

    except Exception as e:
        print(f"❌ Load failed: {e}")
        return False


def mark_attendance(name, confidence=None):
    return db_mark_attendance(name, confidence)


# ====================== LIVE STREAM ======================
def generate_frames():
    global video, attendance_marked, recognized_person, stop_start_time

    # Reset flags every time a new stream starts
    attendance_marked = False
    recognized_person = None
    stop_start_time = None

    if model is None:
        if not load_model():
            error = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(error, "MODEL / CAMERA ERROR", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            while True:
                _, buf = cv2.imencode('.jpg', error)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(0.5)
            return

    print("🚀 Live stream started...")

    while True:
        if video is None or not video.isOpened():
            break

        success, frame = video.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

        name = "Unknown"
        conf = 999

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.equalizeHist(roi)
            roi = cv2.resize(roi, (200, 200))

            try:
                label, conf = model.predict(roi)
                # Stricter threshold (lower = better match)
                name = labels.get(label, "Unknown") if conf < 85 else "Unknown"
            except Exception:
                name = "Unknown"
                conf = 999

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({conf:.1f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name != "Unknown" and not attendance_marked:
                if mark_attendance(name, confidence=conf):
                    attendance_marked = True
                    recognized_person = name
                    stop_start_time = time.time()
                    print(f"🎉 {name} recognized")

        # Success overlay
        if attendance_marked and recognized_person and stop_start_time:
            cv2.putText(frame, "ATTENDANCE MARKED!", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
            cv2.putText(frame, recognized_person, (90, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(frame, "Redirecting...", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if time.time() - stop_start_time > 5:
                print("🛑 Redirecting to home...")
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.3)
                return

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)


# ====================== ROUTES ======================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start')
def start():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == "admin" and password == "admin123":
            session['admin_logged_in'] = True
            return redirect(url_for('analytics'))
        else:
            error = "Invalid username or password"
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))


@app.route('/attendance')
@login_required
def attendance():
    try:
        records = get_all_attendance(limit=200)
        if records:
            df = pd.DataFrame(records)
            df = df[['name', 'timestamp', 'confidence']]
            df.columns = ['Name', 'Timestamp', 'Confidence']
            df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1f}" if x else "N/A")
            table = df.to_html(classes="table table-hover table-striped table-bordered",
                               index=False, border=0)
        else:
            table = "<div class='alert alert-info'>No attendance records found yet.</div>"
    except Exception as e:
        table = f"<div class='alert alert-danger'>Error: {e}</div>"

    return render_template("attendance.html", table=table)


@app.route('/analytics')
@login_required
def analytics():
    try:
        stats = get_attendance_stats()
    except Exception as e:
        stats = None
        print("Analytics error:", e)
    return render_template("analytics.html", stats=stats)


@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    message = None
    message_type = None

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        if not name:
            message = "Please enter a student name"
            message_type = "danger"
        else:
            os.makedirs(os.path.join("dataset", name), exist_ok=True)
            success, msg = db_add_student(name)
            if success:
                return redirect(url_for('capture_page', name=name))
            else:
                message = msg
                message_type = "danger"

    return render_template("add_student.html", message=message, message_type=message_type)


@app.route('/capture/<name>')
@login_required
def capture_page(name):
    folder = os.path.join("dataset", name)
    count = 0
    if os.path.exists(folder):
        count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
    return render_template("capture.html", name=name, count=count)


@app.route('/save_face/<name>', methods=['POST'])
@login_required
def save_face(name):
    global face_cascade

    try:
        # Force load the cascade if it is missing
        if face_cascade is None:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("✅ Face cascade loaded inside save_face")

        data = request.get_json()

        if not data or 'image' not in data:
            return {"success": False, "message": "No image data received", "count": 0}

        # Remove the header part of base64
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"success": False, "message": "Failed to decode image", "count": 0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected. Please look at the camera.",
                "count": 0
            }

        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        roi = gray[y:y+h, x:x+w]
        roi = cv2.equalizeHist(roi)
        roi = cv2.resize(roi, (200, 200))

        folder = os.path.join("dataset", name)
        os.makedirs(folder, exist_ok=True)

        existing = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
        filename = os.path.join(folder, f"{name}_{existing+1:03d}.jpg")
        cv2.imwrite(filename, roi)

        return {
            "success": True,
            "message": f"Saved image {existing + 1}",
            "count": existing + 1
        }

    except Exception as e:
        print("Save face error:", e)
        return {"success": False, "message": str(e), "count": 0}
    
def save_face_image(name):
    global video, face_cascade

    try:
        if video is None or not video.isOpened():
            if not load_model():
                return False, "Camera not available"

        success, frame = video.read()
        if not success:
            return False, "Failed to read frame from camera"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

        if len(faces) == 0:
            return False, "No face detected. Please look directly at the camera."

        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        roi = gray[y:y+h, x:x+w]
        roi = cv2.equalizeHist(roi)
        roi = cv2.resize(roi, (200, 200))

        folder = os.path.join("dataset", name)
        os.makedirs(folder, exist_ok=True)

        existing = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
        filename = os.path.join(folder, f"{name}_{existing+1:03d}.jpg")

        cv2.imwrite(filename, roi)
        return True, f"Saved image {existing + 1}"

    except Exception as e:
        return False, f"Error: {str(e)}"


@app.route('/train_model', methods=['GET', 'POST'])
@login_required
def train_model():
    message = None
    message_type = None

    if request.method == 'POST':
        success, msg = train_face_model()
        message = msg
        message_type = "success" if success else "danger"

        if success:
            # Reload model into memory
            global model, labels
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read("face_model.yml")
            labels = np.load("labels.npy", allow_pickle=True).item()
            print("✅ Model reloaded")

    return render_template("train_model.html", message=message, message_type=message_type)

@app.route("/recognize", methods=["POST"])
def recognize():
    global model, labels, face_cascade

    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    if model is None and os.path.exists("face_model.yml") and os.path.exists("labels.npy"):
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read("face_model.yml")
        labels = np.load("labels.npy", allow_pickle=True).item()
    if model is None:
        return {"marked": False, "name": None, "message": "Face model was not uploaded"}

    data = request.get_json(silent=True) or {}
    raw = data.get("image", "")
    if "," not in raw:
        return {"marked": False, "name": None, "message": "No frame received"}

    frame = cv2.imdecode(
        np.frombuffer(base64.b64decode(raw.split(",")[1]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    if frame is None:
        return {"marked": False, "name": None, "message": "Bad frame"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 6)
    if len(faces) == 0:
        return {"marked": False, "name": None, "message": "No face detected"}

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    roi = cv2.resize(cv2.equalizeHist(gray[y:y + h, x:x + w]), (200, 200))
    label, conf = model.predict(roi)
    name = labels.get(label, "Unknown") if conf < 85 else "Unknown"
    if name == "Unknown":
        return {"marked": False, "name": None, "message": f"Unknown face ({conf:.0f})"}
    if mark_attendance(name, confidence=conf):
        return {"marked": True, "name": name, "message": "Marked"}
    return {"marked": True, "name": name, "message": "Already marked"}

def train_face_model():
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        return False, "Dataset folder not found"

    images = []
    labels_list = []
    label_map = {}
    label_id = 0
    total_images = 0

    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_map[label_id] = person_name
        person_images = 0

        for file in os.listdir(person_folder):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(person_folder, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels_list.append(label_id)
            person_images += 1
            total_images += 1

        print(f"  → {person_name}: {person_images} images")
        label_id += 1

    if total_images == 0:
        return False, "No images found. Please capture faces first."

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.array(labels_list))
    model.write("face_model.yml")
    np.save("labels.npy", label_map)

    return True, f"Training complete! {len(label_map)} people, {total_images} images."


# ====================== START ======================
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)