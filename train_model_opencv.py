import cv2
import numpy as np
import os

print("🔥 Face Training Tool (Fixed Preprocessing)")
print("===========================================")

name = input("Enter the person's FULL name (exactly as before): ").strip()
if not name:
    print("❌ Name cannot be empty!")
    exit()

person_folder = f"dataset/{name}"
os.makedirs(person_folder, exist_ok=True)

print(f"\n📸 Capturing more photos for → {name}")
print("Move your head slowly (left/right/up/down) + try different expressions")
print("Press 'Q' when done\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
while count < 150:          # ← Increased to 150 for better accuracy
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    for (x, y, w, h) in faces:
        # EXACT same preprocessing as your live code
        roi = gray[y:y + h, x:x + w]
        roi = cv2.equalizeHist(roi)          # ← This was missing!
        roi = cv2.resize(roi, (200, 200))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing {count+1}/150", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(f"{person_folder}/{name}_{count:03d}.jpg", roi)
        count += 1

    cv2.imshow("Training - Press Q to stop", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 150:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Captured {count} images for {name}")

# ====================== TRAIN ======================
print("🔄 Training model with correct preprocessing...")

images = []
labels_list = []
label_map = {}
label_id = 0

for person_name in os.listdir("dataset"):
    folder = f"dataset/{person_name}"
    if not os.path.isdir(folder):
        continue

    label_map[label_id] = person_name
    for file in os.listdir(folder):
        img = cv2.imread(f"{folder}/{file}", cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels_list.append(label_id)
    label_id += 1

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, np.array(labels_list))

model.write("face_model.yml")
np.save("labels.npy", label_map)

print("🎉 TRAINING COMPLETE!")
print(f"   Trained {len(label_map)} people")
print("   Now restart your attendance_system.py")