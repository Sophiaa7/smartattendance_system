from flask import Flask, render_template, Response, redirect, url_for
from attendance_system import generate_frames
from database import get_all_attendance, get_attendance_by_date, init_db
import pandas as pd
from database import get_all_attendance, get_attendance_by_date, init_db, get_attendance_stats


app = Flask(__name__)

@app.route('/analytics')
def analytics():
    try:
        stats = get_attendance_stats()
    except Exception as e:
        stats = None
        error = str(e)

    return render_template("analytics.html", stats=stats)
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    return redirect(url_for('video_feed'))


@app.route('/attendance')
def attendance():
    try:
        records = get_all_attendance(limit=200)

        if records:
            df = pd.DataFrame(records)
            df = df[['name', 'timestamp', 'confidence']]
            df.columns = ['Name', 'Timestamp', 'Confidence']
            # Format confidence nicely
            df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1f}" if x else "N/A")
            table = df.to_html(classes="table table-hover table-striped table-bordered align-middle", 
                               index=False, border=0, justify="center")
        else:
            table = "<div class='alert alert-info'>No attendance records found yet.</div>"

    except Exception as e:
        table = f"<div class='alert alert-danger'>Error: {e}</div>"

    return render_template("attendance.html", table=table)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)