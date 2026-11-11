import cv2
import os
import time
from datetime import datetime

# ====================== CONFIG ======================
name = input("Enter person's name (e.g., Yinka): ").strip()

dataset_path = os.path.join("dataset", name)
os.makedirs(dataset_path, exist_ok=True)

print(f"📁 Images will be saved to: {os.path.abspath(dataset_path)}")
print("Instructions:")
print("   - Look at the camera and slowly move your head (left, right, up, down)")
print("   - Show different expressions (neutral, smile, talking)")
print("   - Press 's' to start auto-capture")
print("   - Press 'q' or ESC to quit early\n")

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
max_images = 60          # Change this if you want more/less
auto_capturing = False
last_capture_time = 0

print("Press 's' to begin auto-capturing...")

while count < max_images:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    display_frame = frame.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if auto_capturing and (time.time() - last_capture_time) > 0.4:  # Capture every 0.4 seconds
            # Crop and preprocess the face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)   # Better for training

            filename = os.path.join(dataset_path, f"{name}_{count:03d}_{datetime.now().strftime('%H%M%S')}.jpg")
            success = cv2.imwrite(filename, face_roi)

            if success:
                print(f"✅ Saved image {count+1}/{max_images}")
                count += 1
                last_capture_time = time.time()
            else:
                print("Failed to save image")

    # Show count on screen
    cv2.putText(display_frame, f"Images: {count}/{max_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Data Collector - Press 's' to start | 'q' to quit", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not auto_capturing:
        auto_capturing = True
        print("🚀 Auto-capturing started... Move your head slowly!")

    elif key == ord('q') or key == 27:   # ESC key
        print("Exiting...")
        break

video.release()
cv2.destroyAllWindows()

print(f"\n✅ Capture finished! Total images saved: {count}")
print(f"Folder: {os.path.abspath(dataset_path)}")